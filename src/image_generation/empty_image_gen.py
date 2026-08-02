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

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

FILE_PATH = empty_dataset_path

RESOLUTION = 256
TIME_STEP = 5
CHANNELS = 2
N_WORKERS, CHUNKSIZE = 6, 2

def run_one_sample(args):
    X, Y, Z = args
    tdi_dict = make_tdi_dict(X, Y, Z, whiten=False)
    
    tcen_arr = np.random.uniform(-3, 3, size=TIME_STEP)*3600
    
    images = np.zeros((len(tcen_arr), RESOLUTION, RESOLUTION, CHANNELS))
    t_axes = np.zeros((len(tcen_arr), RESOLUTION, CHANNELS))
    f_axes = np.zeros((len(tcen_arr), RESOLUTION, CHANNELS))
    tcen_vals = np.zeros(len(tcen_arr))

    for i, tcen in enumerate(tcen_arr):
        event = PIPE['t_inj'] - tcen
        QT = np.zeros((RESOLUTION, RESOLUTION, CHANNELS))
        t_axis = np.zeros((RESOLUTION, CHANNELS))
        f_axis = np.zeros((RESOLUTION, CHANNELS))
        for j, spec in enumerate([SPECS['q6'], SPECS['q16']]):
            t_arr, f_arr, data = generate_qscan(tdi_dict, 'A', PIPE, event, resolution=RESOLUTION,
                                frange=spec['frange'], trange=spec['trange'], Q=spec['Q'])

            QT[:, :, j] = data
            t_axis[:, j] = t_arr
            f_axis[:, j] = f_arr
        
        images[i], t_axes[i], f_axes[i], tcen_vals[i] = QT, t_axis, f_axis, tcen

    return images, t_axes, f_axes, tcen_vals

def run_chunk(job_chunk):
    results = []
    for job in job_chunk:
        try:
            results.append(run_one_sample(job))
        except Exception as e:
            print(f"Error processing job {job}: {e}")
            continue
    return results

def chunkify(lst, chunksize):
    for i in range(0, len(lst), chunksize):
        yield lst[i:i + chunksize]

def main():
    start = time.time()

    keys = ['QT', 't_qscan', 'f_qscan']

    h5file = h5py.File(FILE_PATH, "r+")
    if keys[0] not in h5file:
        h5file.close()
        h5file = create_imageset(FILE_PATH, TIME_STEP, RESOLUTION, CHANNELS, keys=keys)
        X_array = h5file["X"][:]
        Y_array = h5file["Y"][:]
        Z_array = h5file["Z"][:]

    else:
        n_images = h5file[keys[0]].shape[0]
        n_signals = h5file["X"].shape[0]

        # If all signals already have images, exit early
        if n_images >= n_signals:
            print("All signals already have generated images. Nothing to do.")
            h5file.close()
            return

        # Otherwise continue from where we left off
        X_array = h5file["X"][n_images:]
        Y_array = h5file["Y"][n_images:]
        Z_array = h5file["Z"][n_images:]

    # Prepare job list
    jobs = [(X_array[i], Y_array[i], Z_array[i])
             for i in range(len(X_array))]

    print(f"Running with {N_WORKERS} workers")

    job_chunks = list(chunkify(jobs, CHUNKSIZE))

    total_chunks = len(job_chunks)
    total_samples = len(jobs)

    completed_samples = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(run_chunk, chunk) for chunk in job_chunks]

        for future in tqdm(futures, total=total_chunks, desc="Chunks completed"):
            results = future.result()
            
            completed_samples += len(results)
            tqdm.write(f"Samples done: {completed_samples}/{total_samples}")
            
            for images, t_axes, f_axes, tcen_vals in results:
                append_image(h5file, images, t_axes, f_axes, tcen_vals, keys=keys)
    
    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()