# Generate simulated instrumental-glitch samples for the glitch dataset class.
import os
# Limit numerical-library threading so each worker uses one CPU thread.
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

from helpers.simulation import run_simulation, run_tdi
from helpers.h5file_helpers import *
from helpers.config import *
from scipy.stats import truncnorm

from joblib import load
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

@contextlib.contextmanager
def suppress_output():
    # Temporarily silence verbose simulator output within a worker process.
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

with h5py.File(orbits_path, 'r') as orb:
    # Read the time span available in the precomputed orbit file.
    ORB_TO = orb.attrs['t0']
    ORB_SIZE = orb.attrs['size']
    ORB_DT = orb.attrs['dt']

LOG_BETA_MIN, LOG_BETA_MAX = 0, np.log10(5e3)
# Define the glitch-parameter sampling ranges in log space.
LOG_AMP_MIN, LOG_AMP_MAX = -14, -11
INJ_POINTS = ['tm_12', 'tm_23', 'tm_13',
              'tm_21', 'tm_32', 'tm_31']

N_WORKERS, CHUNKSIZE = 6, 2

# Switch between the smaller test set and the production dataset.
TEST = False
if TEST:
    FILE_PATH = glitch_testset_path
    NSAMPLES = 500
else:
    FILE_PATH = glitch_dataset_path
    NSAMPLES = 80

def run_one_sample(args):
    # Construct, simulate, and transform one injected shapelet glitch.
    amp, beta, inj_point, t0, pipe = args

    pipe['t0'] = t0
    glitches = [{'type':'IntegratedShapeletGlitch', 't_inj': pipe['t_inj'], 'beta':beta,
             'level':2*amp*beta, 'inj_point':inj_point}]
    gws = []

    pid = os.getpid()
    local_sim_path = f"{simulation_path}_{pid}.h5"
    local_gw_path  = None
    local_glitch_path = f"{glitch_path}_{pid}.h5"

    with suppress_output():
        run_simulation(
            gws, glitches, pipe,
            local_gw_path, local_glitch_path, orbits_path,
            local_sim_path,
            disable_noise=PIPE['keep_noises'])
        tdi_dict = run_tdi(local_sim_path, pipe)

    os.remove(local_sim_path)
    os.remove(local_glitch_path)

    return tdi_dict, amp, beta, inj_point, t0

def run_chunk(job_chunk):
    # Keep failures local to individual jobs within a worker chunk.
    results = []
    for job in job_chunk:
        try:
            results.append(run_one_sample(job))
        except Exception as e:
            print(f"Error processing job {job}: {e}")
            continue
    return results

def chunkify(lst, chunksize):
    # Yield fixed-size job groups for submission to the process pool.
    for i in range(0, len(lst), chunksize):
        yield lst[i:i + chunksize]

def get_filtered_params(nsamples, forest_path, lower, upper):
    # Retain candidate log-parameters that the trained forest rates as usable.
    xmin, ymin = lower
    xmax, ymax = upper

    samples = []
    clf = load(forest_path)
    while len(samples) < nsamples:
        x = np.random.uniform(xmin, xmax, nsamples)
        y = np.random.uniform(ymin, ymax, nsamples)

        X_new = np.column_stack((x, y))
        prob = clf.predict_proba(X_new)[:, 1]
        samples.extend(X_new[prob > 0.9])

    samples = np.vstack(samples)[:nsamples]
    return samples

def get_param_values(nsamples):
    # Draw glitch parameters, injection points, and valid orbit start times.
    if TEST:
        beta_array = 10**np.random.uniform(LOG_BETA_MIN, LOG_BETA_MAX, size=nsamples)
        amp_array  = 10**np.random.uniform(LOG_AMP_MIN, LOG_AMP_MAX, size=nsamples)
    else:
        glitch_params = get_filtered_params(nsamples, glitch_forest_path, 
                                            lower=[LOG_BETA_MIN, LOG_AMP_MIN],
                                            upper=[LOG_BETA_MAX, LOG_AMP_MAX])
        beta_array = 10**glitch_params[:,0]
        amp_array = 10**glitch_params[:,1]
         
    inj_point_array = np.random.choice(INJ_POINTS, nsamples)
    t0_array = ORB_TO + np.random.uniform(0.01, 0.99, size=nsamples)*ORB_SIZE*ORB_DT

    return amp_array, beta_array, inj_point_array, t0_array

def main():
    # Create the output dataset and populate it with parallel simulations.
    start = time.time()

    amp_array, beta_array, inj_point_array, t0_array = get_param_values(NSAMPLES)

    jobs = [(amp_array[i], beta_array[i], inj_point_array[i], t0_array[i], PIPE) for i in range(NSAMPLES)]
    
    # Create dataset
    #if os.path.exists(FILE_PATH):
     #   os.remove(FILE_PATH)

    # Initialise the destination HDF5 structure before appending samples.
    h5file = create_glitch_dataset(FILE_PATH, PIPE['size'])

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

            for tdi_dict, amp, beta, inj_point, t0 in results:
                append_glitch_sample(h5file, tdi_dict, amp, beta, inj_point, t0)
    
    #sort_glitch_dataset(h5file)
    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
