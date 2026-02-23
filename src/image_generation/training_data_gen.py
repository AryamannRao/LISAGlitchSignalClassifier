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
from config import *

KEY = 'QT'

def generate_dataset(loc, paths):
    gw_path, glitch_path, mixed_path, empty_path = paths

    if not os.path.exists(loc):
        with h5py.File(gw_path, "r") as f:
            gw_images = f[KEY][:]

        with h5py.File(glitch_path, "r") as f:
            glitch_images = f[KEY][:]

        with h5py.File(mixed_path, "r") as f:
            mixed_images = f[KEY][:]

        with h5py.File(empty_path, "r") as f:
            noise_images = f[KEY][:]
        
        gw_labels = np.tile([1,0], (len(gw_images), 1))
        glitch_labels = np.tile([0,1], (len(glitch_images), 1))
        mixed_labels = np.tile([1,1], (len(mixed_images), 1))
        noise_labels = np.tile([0,0], (len(noise_images), 1))

        images = np.concatenate([gw_images, glitch_images, mixed_images, noise_images], axis=0)
        labels = np.concatenate([gw_labels, glitch_labels, mixed_labels, noise_labels], axis=0)

        perm = np.random.permutation(len(images))
        images = images[perm]
        labels = labels[perm]

        with h5py.File(loc, "w") as f:
            f.create_dataset("images", data=images)
            f.create_dataset("labels", data=labels)
    else:
        print(f"Dataset already exists at {loc}")

def main():
    generate_dataset(training_lm_dataset_path, 
                        (gw_lm_dataset_path, glitch_lm_dataset_path, mixed_lm_dataset_path, empty_lm_dataset_path))
    generate_dataset(training_hm_dataset_path, 
                        (gw_hm_dataset_path, glitch_hm_dataset_path, mixed_hm_dataset_path, empty_hm_dataset_path))
    generate_dataset(training_ehm_dataset_path, 
                        (gw_ehm_dataset_path, glitch_ehm_dataset_path, mixed_ehm_dataset_path, empty_ehm_dataset_path))

if __name__ == "__main__":
    main()