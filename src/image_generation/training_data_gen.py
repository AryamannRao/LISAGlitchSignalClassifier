import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
import time
from pathlib import Path

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

            img_ds = f.create_dataset(
                "images",
                shape=(total,256,256,2),
                dtype="float32",
                compression="lzf",
                chunks=(64,256,256,2)
            )

            label_ds = f.create_dataset(
                "labels",
                shape=(total,2),
                dtype="int8"
            )

            sim_index_ds = f.create_dataset(
                "sim_index",
                shape=(total,),
                dtype="int64"
            )

            time_index_ds = f.create_dataset(
                "time_index",
                shape=(total,),
                dtype="int64"
            )

            idx = 0

            for path, label in [
                (gw_path, [1,0]),
                (glitch_path, [0,1]),
                (mixed_path, [1,1]),
                (empty_path, [0,0]),
            ]:

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

def main():
    start = time.time()
    generate_dataset(training_dataset_path, 
                        (gw_dataset_path, glitch_dataset_path, mixed_dataset_path, empty_dataset_path))
    end = time.time()
    print(f"Training dataset generation took {end - start:.2f} seconds")

if __name__ == "__main__":
    main()