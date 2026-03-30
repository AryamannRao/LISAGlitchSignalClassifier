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

from helpers.simulation import run_simulation, run_tdi
from helpers.h5file_helpers import *
from helpers.config import *
from scipy.stats import truncnorm

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

NSAMPLES = 666
BETA_MIN = 1
AMP_MIN, AMP_MAX = 1e-16, 1e-10
INJ_POINTS = ['tm_12', 'tm_23', 'tm_13',
              'tm_21', 'tm_32', 'tm_31']

N_WORKERS, CHUNKSIZE = 6, 2

def run_one_sample(args):
    amp, beta, inj_point, pipe = args

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

    return tdi_dict, amp, beta, inj_point

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

def truncated_2d_gaussian(mu, cov, xmin, size):
    mu_x, mu_y = mu
    
    # Step 1: sample x (vectorized)
    sigma_x = np.sqrt(cov[0, 0])
    a = (xmin - mu_x) / sigma_x
    
    x = truncnorm.rvs(a, np.inf, loc=mu_x, scale=sigma_x, size=size)
    
    # Step 2: compute conditional mean (vectorized)
    beta = cov[1, 0] / cov[0, 0]
    mu_cond = mu_y + beta * (x - mu_x)
    
    # Step 3: conditional variance (scalar)
    var_cond = cov[1, 1] - (cov[1, 0]**2) / cov[0, 0]
    sigma_cond = np.sqrt(var_cond)
    
    # Step 4: sample y (vectorized)
    y = np.random.normal(mu_cond, sigma_cond, size=size)
    
    return np.column_stack((x, y))

def get_param_values(nsamples):
    params = np.loadtxt(lpf_ord_param_path, skiprows=1)

    lpf_betas, lpf_levels = params[:, 0], np.abs(params[:, 1])
    lpf_levels = lpf_levels[lpf_betas > BETA_MIN]
    lpf_betas = lpf_betas[lpf_betas > BETA_MIN]

    lpf_betas = lpf_betas[(lpf_levels > AMP_MIN) & (lpf_levels < AMP_MAX)]
    lpf_levels = lpf_levels[(lpf_levels > AMP_MIN) & (lpf_levels < AMP_MAX)]

    params = np.vstack((np.log10(lpf_betas), np.log10(lpf_levels))).T

    mu = np.mean(params, axis=0)
    cov = np.cov(params, rowvar=False)
    samples = truncated_2d_gaussian(mu, cov, xmin=np.log10(BETA_MIN), size=nsamples)
    
    beta_array = 10**samples[:,0]
    amp_array  = 10**samples[:,1]
    inj_point_array = np.random.choice(INJ_POINTS, nsamples)

    return amp_array, beta_array, inj_point_array

def main():
    start = time.time()

    amp_array, beta_array, inj_point_array = get_param_values(NSAMPLES)

    jobs = [(amp_array[i], beta_array[i], inj_point_array[i], PIPE) for i in range(NSAMPLES)]
    
    # Create dataset
    if os.path.exists(glitch_dataset_path):
        os.remove(glitch_dataset_path)

    h5file = create_glitch_dataset(glitch_dataset_path, PIPE['size'])

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

            for tdi_dict, amp, beta, inj_point in results:
                append_glitch_sample(h5file, tdi_dict, amp, beta, inj_point)
    
    #sort_glitch_dataset(h5file)
    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()