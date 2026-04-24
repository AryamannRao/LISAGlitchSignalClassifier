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

MASS_CATEGORY = 'high_mass'  # 'low_mass', 'high_mass', 'ext_high_mass'
SPEC = SPECS[MASS_CATEGORY]

if MASS_CATEGORY == 'low_mass':
    FILE_PATH = gw_lm_dataset_path
elif MASS_CATEGORY == 'high_mass':
    FILE_PATH = gw_hm_dataset_path
elif MASS_CATEGORY == 'ext_high_mass':
    FILE_PATH = gw_ehm_dataset_path

RESOLUTION = 256
TIME_STEP = 5
N_WORKERS, CHUNKSIZE = 6, 2
REPRESENTATION = 'qscan'  # 'qscan' or 'varqscan'

def run_one_sample(args):
    tcen0, X, Y, Z = args
    tdi_dict = make_tdi_dict(X, Y, Z)
    
    tcen_arr = np.linspace(-2, 2, TIME_STEP)*3600 + tcen0
    
    images = np.zeros((len(tcen_arr), RESOLUTION, RESOLUTION, 3))
    t_axes = np.zeros((len(tcen_arr), RESOLUTION))
    f_axes = np.zeros((len(tcen_arr), RESOLUTION))
    tcen_vals = np.zeros(len(tcen_arr))

    for i, tcen in enumerate(tcen_arr):
        event = {'t_inj': PIPE['t_inj'] - tcen}
        QT = np.zeros((RESOLUTION, RESOLUTION, 3))
        for channel in ['A', 'E', 'T']:
            if REPRESENTATION == 'varqscan':
                t_arr, f_arr, data = varq_transform(tdi_dict, channel, PIPE, event, resolution=RESOLUTION,
                                            frange=SPEC['frange'], trange=SPEC['trange'], Qvals=SPEC['Qvals'])
            elif REPRESENTATION == 'qscan':
                t_arr, f_arr, data = generate_qscan(tdi_dict, channel, PIPE, event, resolution=RESOLUTION,
                                frange=SPEC['frange'], trange=SPEC['trange'], Q=SPEC['Q'])

            QT[:, :, ['A', 'E', 'T'].index(channel)] = data
        
        images[i], t_axes[i], f_axes[i], tcen_vals[i] = QT, t_arr, f_arr, tcen

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

    if REPRESENTATION == 'varqscan':
        keys = ['varQT', 't_varq', 'f_varq']
    elif REPRESENTATION == 'qscan':
        keys = ['QT', 't_qscan', 'f_qscan']

    h5file = h5py.File(FILE_PATH, "r+")
    if keys[0] not in h5file:
        h5file.close()
        h5file = create_imageset(FILE_PATH, TIME_STEP, RESOLUTION, keys=keys)
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
        n = n_images
        X_array = h5file["X"][n:]
        Y_array = h5file["Y"][n:]
        Z_array = h5file["Z"][n:]

    tcen_start = np.random.uniform(-1, 1, size=len(X_array))*3600

    # Prepare job list
    jobs = [(tcen_start[i], X_array[i], Y_array[i], Z_array[i])
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