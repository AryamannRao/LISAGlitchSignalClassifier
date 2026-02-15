import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import contextlib
import numpy as np
import time

from simulation import run_simulation, run_tdi
from h5file_helpers import create_dataset, append_gw_sample
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

def run_one_sample(args):
    level, beta, inj_point, pipe = args

    gws = [{'type':'ReducedOneSidedDoubleExpGW', 't_inj': 0, 'amp':0, 't_rise':1, 't_fall':1}]

    glitches = [{'type':'ShapeletGlitch', 't_inj': pipe['t_inj'], 'beta':beta,
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

    return tdi_dict, level, beta, inj_point

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
    levels = np.random.uniform(1e-22, 1e-20, nsamples)
    betas = np.random.uniform(0.5, 10, nsamples)
    inj_points = np.random.choice(['tm_12', 'tm_23', 'tm_13'], nsamples)
    return levels, betas, inj_points

def main():
    start = time.time()

    nsamples = 1000
    levels, betas, inj_points = get_param_values(nsamples)

    pipe = {'t0': 10368000, 'dt': 0.25,'size': 10 * 3600 / 0.25, 't_inj': 5 * 3600}

    jobs = [(levels[i], betas[i], inj_points[i], pipe) for i in range(nsamples)]

    # Create dataset
    if os.path.exists(glitch_dataset_path):
        os.remove(glitch_dataset_path)

    h5file = create_dataset(glitch_dataset_path, pipe['size'])

    n_workers = 6
    print(f"Running with {n_workers} workers")

    chunksize = 2  # start with 2 or 3
    job_chunks = list(chunkify(jobs, chunksize))

    total_chunks = len(job_chunks)
    total_samples = len(jobs)

    completed_samples = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_chunk, chunk) for chunk in job_chunks]

        for future in tqdm(as_completed(futures), total=total_chunks, desc="Chunks completed"):
            results = future.result()
            
            completed_samples += len(results)
            tqdm.write(f"Samples done: {completed_samples}/{total_samples}")

            for tdi_dict, level, beta, inj_point in results:
                append_gw_sample(h5file, tdi_dict, level, beta, inj_point)
    
    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
