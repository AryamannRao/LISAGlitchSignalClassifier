import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import contextlib
import numpy as np
import time
from pathlib import Path
from scipy.stats import truncnorm

# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.simulation import *
from helpers.h5file_helpers import *
from helpers.config import *

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

with h5py.File(orbits_path, 'r') as orb:
    ORB_TO = orb.attrs['t0']
    ORB_SIZE = orb.attrs['size']
    ORB_DT = orb.attrs['dt']

NSAMPLES = 100
MASS_CATEGORY = 'ext_high_mass'

if MASS_CATEGORY == 'low_mass':
    FILE_PATH = mixed_lm_dataset_path
    LOG_MASS_MIN, LOG_MASS_MAX = 4, 5
elif MASS_CATEGORY == 'high_mass':
    FILE_PATH = mixed_hm_dataset_path
    LOG_MASS_MIN, LOG_MASS_MAX = 5, 6
elif MASS_CATEGORY == 'ext_high_mass':
    FILE_PATH = mixed_ehm_dataset_path
    FILE_PATH = MIXED_DATASETS / 'mixed_gwbeta_way.h5'
    LOG_MASS_MIN, LOG_MASS_MAX = 6, 7

Q_MIN, Q_MAX = 1, 5
LOG_D_MIN, LOG_D_MAX = 3, 5
SPIN_MIN, SPIN_MAX = -0.99, 0.99

BETA_MIN = 1
AMP_MIN, AMP_MAX = 1e-16, 1e-10
INJ_POINTS = ['tm_12', 'tm_23', 'tm_13',
              'tm_21', 'tm_32', 'tm_31']

SEP_MIN, SEP_MAX = -1.5, 1.5

N_WORKERS, CHUNKSIZE = 6, 2

def set_seed(seed):
    np.random.seed(seed)

def run_one_sample(args):
    m1, m2, d, spin1, spin2, iota, gw_beta, gw_lambda,\
          amp, beta, inj_point, sep, t0, pipe = args

    pipe['t0'] = t0
    gws = [{'type': 'BinaryInspiralGW',
        'm1': m1, 'm2': m2, 'd': d,
        'spin1': spin1, 'spin2': spin2, 'iota': iota,
        'gw_beta': gw_beta, 'gw_lambda': gw_lambda,
        't_inj': pipe['t_inj'] + sep * 3600, 'domain': 'freq'}]

    glitches = [{'type':'IntegratedShapeletGlitch', 't_inj': pipe['t_inj'] - sep * 3600, 'beta':beta,
             'level':2*amp*beta, 'inj_point':inj_point}]

    pid = os.getpid()
    local_sim_path = f"{simulation_path}_{pid}.h5"
    local_gw_path  = f"{gw_path}_{pid}.h5"
    local_glitch_path = f"{glitch_path}_{pid}.h5"

    with suppress_output():
        run_simulation(
            gws, glitches, pipe,
            local_gw_path, local_glitch_path, orbits_path,
            local_sim_path,
            disable_noise=pipe['keep_noises']
        )
        tdi_dict = run_tdi(local_sim_path, pipe)

    os.remove(local_sim_path)
    os.remove(local_gw_path)
    os.remove(local_glitch_path)

    return tdi_dict, m1, m2, d, spin1, spin2, iota,\
        gw_beta, gw_lambda, amp, beta, inj_point, sep, t0

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
    Mc_array = 10**np.random.uniform(LOG_MASS_MIN, LOG_MASS_MAX, size=nsamples)
    q_array = np.random.uniform(Q_MIN, Q_MAX, size=nsamples)

    m1_array = Mc_array * ((1 + q_array)**(1/5) / q_array**(3/5))
    m2_array = m1_array * q_array
    spin1_array = np.random.uniform(0, 0.99, size=nsamples)
    spin2_array = spin1_array * np.random.choice([-1, 1], size=nsamples)
    d_array = 10**np.random.uniform(LOG_D_MIN, LOG_D_MAX, size=nsamples)

    iota_array = np.arccos(np.random.uniform(-1, 1, size=nsamples))
    gw_beta_array = np.arcsin(np.random.uniform(-1, 1, size=nsamples))
    #iota_array = np.zeros(shape=nsamples)
    #gw_beta_array = np.random.uniform(-np.pi/2, np.pi/2, size=nsamples)
    gw_lambda_array = np.random.uniform(0, 2*np.pi, size=nsamples)

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

    #beta_array = 10**np.random.uniform(0, np.log10(5e3), size=nsamples)
    #amp_array  = 10**np.random.uniform(-14, -11, size=nsamples)
    inj_point_array = np.random.choice(INJ_POINTS, nsamples)

    sep_array = np.random.uniform(SEP_MIN, SEP_MAX, nsamples)
    t0_array = ORB_TO + np.random.uniform(0.01, 0.99, size=nsamples)*ORB_SIZE*ORB_DT

    return m1_array, m2_array, d_array,\
          spin1_array, spin2_array, iota_array, gw_beta_array, gw_lambda_array,\
            amp_array, beta_array, inj_point_array, sep_array, t0_array

def main():
    start = time.time()
    set_seed(42)
    m1_array, m2_array, d_array, spin1_array, spin2_array, iota_array, gw_beta_array, gw_lambda_array,\
         amp_array, beta_array, inj_point_array, sep_array, t0_array = get_param_values(NSAMPLES)

    jobs = [(m1_array[i], m2_array[i], d_array[i], 
             spin1_array[i], spin2_array[i], iota_array[i],
             gw_beta_array[i], gw_lambda_array[i],
             amp_array[i], beta_array[i], inj_point_array[i],
             sep_array[i], t0_array[i], PIPE) for i in range(NSAMPLES)]

    #if os.path.exists(mixed_dataset_path):
     #   os.remove(mixed_dataset_path)
    
    h5file = create_mixed_dataset(FILE_PATH, PIPE['size'])

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
            
            for tdi_dict, m1, m2, d, spin1, spin2, iota, gw_beta, gw_lambda, amp, beta, inj_point, sep, t0 in results:
                append_mixed_sample(h5file, tdi_dict, m1, m2, d, spin1, spin2, iota, gw_beta, gw_lambda,
                                    amp, beta, inj_point, sep, t0)

    #sort_mixed_dataset(h5file)
    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()