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

MASS_CATEGORY = 'ext_high_mass'  # 'low_mass', 'high_mass', 'ext_high_mass'
SPEC = SPECS[MASS_CATEGORY]

if MASS_CATEGORY == 'low_mass':
    FILE_PATH = mixed_lm_dataset_path
elif MASS_CATEGORY == 'high_mass':
    FILE_PATH = mixed_hm_dataset_path
elif MASS_CATEGORY == 'ext_high_mass':
    FILE_PATH = mixed_ehm_dataset_path

RESOLUTION = 256
N_WORKERS, CHUNKSIZE = 6, 2
REPRESENTATION = 'qscan'  # 'qscan' or 'varqscan'

def run_one_sample(args):
    tcen, X, Y, Z = args
    tdi_dict = make_tdi_dict(X, Y, Z)
    
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

    return QT, t_arr, f_arr, tcen

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
        h5file = create_imageset(FILE_PATH, RESOLUTION, keys=keys)
        X_array = h5file["X"][:]
        Y_array = h5file["Y"][:]
        Z_array = h5file["Z"][:]
        sep_array = h5file["sep"][:]

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
        sep_array = h5file["sep"][n_images:]

    tmax = SPEC['trange'][1]
    delta = tmax - sep_array - 0.2
    clip = PIPE['t_inj']/3600 - tmax
    delta[delta > clip] = clip
    tcen_array = np.random.uniform(-delta, delta) * 3600

    # Prepare job list
    jobs = [(tcen_array[i], X_array[i], Y_array[i], Z_array[i])
             for i in range(len(tcen_array))]

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
            
            for QT, t_arr, f_arr, tcen in results:
                append_image(h5file, QT, t_arr, f_arr, tcen, keys=keys)

    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()