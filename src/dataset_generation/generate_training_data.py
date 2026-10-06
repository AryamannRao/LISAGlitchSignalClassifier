# Combine class-specific Q-transform images into a labelled training dataset.
import os
# Limit numerical-library threading so each worker uses one CPU thread.
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

RESOLUTION = 128
CHANNELS = 4

def filter_by_snr(dataset_path, label, threshold=8):
    with h5py.File(dataset_path, 'r') as f:
        # Load ALL images
        images = f['images'][:]

        # Load ALL image metadata
        t_axes = f['image_metadata/t_axis'][:]
        f_axes = f['image_metadata/f_axis'][:]

    snrs = np.sqrt(images**2 - 1)

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
            # Locate distinct clusters of above-threshold time-frequency pixels.
            pdf = gaussian_kde(T_opt)(times)
            peaks, _ = find_peaks(pdf, width=1, prominence=0.05)

            if len(peaks) >= sum(label):
                good_idx.append(index)
        except Exception as e:
            print(f"Sample {index}: {e}")

    good_idx = np.array(good_idx)
    return good_idx

def main():
    # Build the merged dataset, then retain only detectable transient examples.
    h5file = h5py.File(training_dataset_path, 'w')
    create_imageset(h5file, RESOLUTION, CHANNELS)

    for n_smbh in [0,1]:
        for n_glitch in [0,1]:
            dataset_path = str(TRAINING_DATASETS/f'dataset_{n_smbh}{n_glitch}.h5')
            good_idx = filter_by_snr(dataset_path, label=[n_smbh,n_glitch], threshold=8)

            with h5py.File(dataset_path, 'r') as src:
                images = src['images'][good_idx]
                labels = src['labels'][good_idx]
                image_metadata = {key: src[f'image_metadata/{key}'][good_idx]
                    for key in ['t_axis', 'f_axis', 'tcen', 'sim_idx', 'time_idx']}

            append_image(h5file, images, labels, image_metadata)

    h5file.close()

if __name__ == "__main__":
    main()
