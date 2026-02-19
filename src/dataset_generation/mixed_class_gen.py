import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import contextlib
import numpy as np
import time
from pathlib import Path

# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.simulation import *
from helpers.h5file_helpers import *
from config import *

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

NSAMPLES = 1000
LOG_MASS_MIN, LOG_MASS_MAX = 4, 7

#LEVEL_MIN, LEVEL_MAX = 1e-6, 1e-5
BETA_MIN, BETA_MAX = 500, 3600
INJ_POINTS = ['tm_12', 'tm_23', 'tm_13',
              'tm_21', 'tm_32', 'tm_31']

SEP_MIN, SEP_MAX = -1.5, 1.5
PIPE = {'t0': 10368000, 'dt': 0.25,'size': 10 * 3600 / 0.25, 't_inj': 5 * 3600}

N_WORKERS, CHUNKSIZE = 6, 2

def run_one_sample(args):
    m1, m2, d, spin1, spin2, gw_beta, gw_lambda, level, beta, inj_point, sep, pipe = args

    gws = [{'type': 'BinaryInspiralGW',
        'm1': m1, 'm2': m2, 'd': d,
        'spin1': spin1, 'spin2': spin2,
        'gw_beta': gw_beta, 'gw_lambda': gw_lambda,
        't_inj': pipe['t_inj'] + sep * 3600, 'domain': 'freq'}]

    glitches = [{'type':'ShapeletGlitch', 't_inj': pipe['t_inj'] - sep * 3600, 'beta':beta,
             'level':level, 'inj_point':inj_point}]

    pid = os.getpid()
    local_sim_path = f"{simulation_path}_{pid}.h5"
    local_gw_path     = f"{gw_path}_{pid}.h5"
    local_glitch_path = f"{glitch_path}_{pid}.h5"

    with suppress_output():
        run_simulation(
            gws, glitches, pipe,
            local_gw_path, local_glitch_path, orbits_path,
            local_sim_path,
            disable_noise=False
        )

    tdi_dict = run_tdi(local_sim_path, pipe)

    os.remove(local_sim_path)
    os.remove(local_gw_path)
    os.remove(local_glitch_path)

    return tdi_dict, m1, m2, d, spin1, spin2, gw_beta, gw_lambda, level, beta, inj_point, sep

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

def get_param_values(nsamples):
    m1_array = 10**np.random.uniform(LOG_MASS_MIN, LOG_MASS_MAX, size=nsamples)
    m2_array = m1_array * np.random.uniform(1, 5, size=nsamples)
    chirp_array = chirp_mass(m1_array, m2_array)

    spin1_array = np.random.uniform(0, 0.9, size=nsamples)
    spin2_array = spin1_array * np.random.choice([-1, 1], size=nsamples)

    low = chirp_array < 1e5
    med = (chirp_array >= 1e5) & (chirp_array <= 1e6)
    high = chirp_array > 1e6

    level_array = np.empty_like(m1_array, dtype=float)
    level_array[low] = 10**np.random.uniform(-6, -5, size=np.sum(low))
    level_array[med] = 10**np.random.uniform(-6, -5, size=np.sum(med))
    level_array[high] = 10**np.random.uniform(np.log10(5e-7), np.log10(5e-6), size=np.sum(high))

    d_array = 10**np.random.uniform(2.5, 4, size=nsamples)
    gw_beta_array = np.random.uniform(-np.pi/2, np.pi/2, size=nsamples)
    gw_lambda_array = np.random.uniform(0, 2*np.pi, size=nsamples)

    beta_array = np.random.uniform(BETA_MIN, BETA_MAX, nsamples)
    inj_point_array = np.random.choice(INJ_POINTS, nsamples)
    sep_array = np.random.uniform(SEP_MIN, SEP_MAX, nsamples)

    return m1_array, m2_array, d_array,\
          spin1_array, spin2_array, gw_beta_array, gw_lambda_array,\
            level_array, beta_array, inj_point_array, sep_array

def main():
    start = time.time()

    m1_array, m2_array, d_array, spin1_array, spin2_array, gw_beta_array, gw_lambda_array,\
         level_array, beta_array, inj_point_array, sep_array = get_param_values(NSAMPLES)

    # Prepare job list
    jobs = [(m1_array[i], m2_array[i], d_array[i], 
             spin1_array[i], spin2_array[i], 
             gw_beta_array[i], gw_lambda_array[i],
             level_array[i], beta_array[i], inj_point_array[i],
             sep_array[i], PIPE) for i in range(NSAMPLES)]

    if os.path.exists(mixed_dataset_path):
        os.remove(mixed_dataset_path)
    
    h5file = create_mixed_dataset(mixed_dataset_path, PIPE['size'])

    print(f"Running with {N_WORKERS} workers")

    job_chunks = list(chunkify(jobs, CHUNKSIZE))

    total_chunks = len(job_chunks)
    total_samples = len(jobs)

    completed_samples = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(run_chunk, chunk) for chunk in job_chunks]

        for future in tqdm(as_completed(futures), total=total_chunks, desc="Chunks completed"):
            results = future.result()
            
            completed_samples += len(results)
            tqdm.write(f"Samples done: {completed_samples}/{total_samples}")
            
            for tdi_dict, m1, m2, d, spin1, spin2, gw_beta, gw_lambda, level, beta, inj_point, sep in results:
                append_mixed_sample(h5file, tdi_dict, m1, m2, d, spin1, spin2, gw_beta, gw_lambda,
                                    level, beta, inj_point, sep)

    sort_mixed_dataset(h5file)
    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()