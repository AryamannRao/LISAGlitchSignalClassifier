import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
import time
from pathlib import Path

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.simulation import *
from helpers.h5file_helpers import *
from helpers.qtransform import *
from helpers.config import *

KEY = 'QT'

def generate_dataset(loc, paths):
    gw_path, glitch_path, mixed_path, empty_path = paths

    if not os.path.exists(loc):
        total = 0
        for path in [gw_path, glitch_path, mixed_path, empty_path]:
            with h5py.File(path, "r") as f:
                shape = f[KEY].shape
                total += shape[0] * shape[1]

        with h5py.File(loc, "w") as f:
            img_ds = f.create_dataset("images", shape=(total,256,256,2),
                dtype="float32", compression="lzf", chunks=(64,256,256,2))

            label_ds = f.create_dataset("labels", shape=(total,2), dtype="int8")
            sim_index_ds = f.create_dataset("sim_index", shape=(total,), dtype="int64")
            time_index_ds = f.create_dataset("time_index", shape=(total,), dtype="int64")

            idx = 0
            for path, label in [(gw_path, [1,0]), (glitch_path, [0,1]),
                (mixed_path, [1,1]), (empty_path, [0,0]),]:

                # load one dataset
                with h5py.File(path, "r") as f_in:
                    arr = f_in[KEY][:]

                # flatten
                flat = arr.reshape(-1,256,256,2)
                sim_indices = np.repeat(np.arange(arr.shape[0]), arr.shape[1])
                time_indices = np.tile(np.arange(arr.shape[1]), arr.shape[0])
                n = flat.shape[0]

                # write
                img_ds[idx:idx+n] = flat
                label_ds[idx:idx+n] = np.tile(label,(n,1))
                sim_index_ds[idx:idx+n] = sim_indices
                time_index_ds[idx:idx+n] = time_indices
                idx += n

                # free memory
                del arr
                del flat
    else:
        print(f"Dataset already exists at {loc}")

def filter_by_snr(dataset_path, training_dict, label, threshold=8):
    with h5py.File(dataset_path, 'r') as h5:
        t_qscan = h5['t_qscan'][:]
        f_qscan = h5['f_qscan'][:]

    idx = np.where(np.all(training_dict['labels'] == label, axis=1))[0]
    images = training_dict['images'][idx]
    sim_idx = training_dict['sim_index'][idx]
    time_idx = training_dict['time_index'][idx]

    snrs = np.sqrt(images**2 - 1)
    t_axes = t_qscan[sim_idx, time_idx, :, :]
    f_axes = f_qscan[sim_idx, time_idx, :, :]

    good_idx = []
    for index in range(images.shape[0]):
        T_opt = []
        for j in range(snrs.shape[-1]):
            snr = snrs[index, :, :, j]
            t_axis = t_axes[index, :, j]
            f_axis = f_axes[index, :, j]
            _, T_grid = np.meshgrid(f_axis, t_axis, indexing="ij")

            T_opt.append(T_grid[snr > threshold])
        
        T_opt = np.concatenate(T_opt)/3600
        if len(T_opt) < 2:
            if sum(label) == 0:
                good_idx.append(index)
            continue
        try:
            times = np.linspace(np.min(t_axes[index])/3600, np.max(t_axes[index])/3600, 250)
            pdf = gaussian_kde(T_opt)(times)
            peaks, _ = find_peaks(pdf, width=1, prominence=0.05)

            if len(peaks) >= sum(label):
                good_idx.append(index)
        except Exception as e:
            print(f"Sample {index}: {e}")

    good_idx = np.array(good_idx)
    return idx[good_idx]

def main():
    start = time.time()
    generate_dataset(training_dataset_path, 
                        (gw_dataset_path, glitch_dataset_path, mixed_dataset_path, empty_dataset_path))
    end = time.time()
    print(f"Training dataset generation took {end - start:.2f} seconds. Filtering started.")

    training_dict = {}
    with h5py.File(training_dataset_path, 'r') as h5:
        for key in list(h5.keys()):
            training_dict[key] = h5[key][:]

    good_gw = filter_by_snr(gw_dataset_path, training_dict, label=[1,0], threshold=8)
    print('GW filtering done')
    good_glitch = filter_by_snr(glitch_dataset_path, training_dict, label=[0,1], threshold=8)
    print('Glitch filtering done')
    good_mixed = filter_by_snr(mixed_dataset_path, training_dict, label=[1,1], threshold=8)
    print('Mixed filtering done')
    good_empty = np.where(np.all(training_dict['labels'] == [0,0], axis=1))[0]
    print('Empty filtering done')

    good_idx = np.concatenate([good_gw, good_glitch, good_mixed, good_empty]).astype(int)
    with h5py.File(training_dataset_path, 'r+') as h5:
        for key in list(h5.keys()):
            data = h5[key][good_idx]
            del h5[key]
            h5.create_dataset(key, data=data)

if __name__ == "__main__":
    main()