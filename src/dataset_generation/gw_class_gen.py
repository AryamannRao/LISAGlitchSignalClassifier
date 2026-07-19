import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import contextlib
import numpy as np
import time
from pathlib import Path

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u

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

NSAMPLES = 1250
MASS_CATEGORY = 'ext_high_mass'

FILE_PATH = gw_dataset_path

LOG_MASS_MIN, LOG_MASS_MAX = 4, 7
Q_MIN, Q_MAX = 1, 5
SPIN_MIN, SPIN_MAX = -0.99, 0.99
Z_MIN, Z_MAX = 0.5, 8

N_WORKERS, CHUNKSIZE = 6, 2

def run_one_sample(args):
    m1, m2, d, spin1, spin2, iota, gw_beta, gw_lambda, t0, pipe = args

    pipe['t0'] = t0
    gws = [{'type': 'BinaryInspiralGW',
        'm1': m1, 'm2': m2, 'd': d,
        'spin1': spin1, 'spin2': spin2, 'iota':iota,
        'gw_beta': gw_beta, 'gw_lambda': gw_lambda,
        't_inj': pipe['t_inj'], 'domain': 'freq'}]

    glitches = []

    pid = os.getpid()
    local_sim_path = f"{simulation_path}_{pid}.h5"
    local_gw_path = f"{gw_path}_{pid}.h5"
    local_glitch_path = None

    with suppress_output():
        run_simulation(
            gws, glitches, pipe,
            local_gw_path, local_glitch_path, orbits_path,
            local_sim_path,
            disable_noise=pipe['keep_noises'], seed=42
        )
        tdi_dict = run_tdi(local_sim_path, pipe)

    os.remove(local_sim_path)
    os.remove(local_gw_path)

    return tdi_dict, m1, m2, d, spin1, spin2, iota, gw_beta, gw_lambda, t0

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
    Mc_array = 10**np.random.uniform(LOG_MASS_MIN, LOG_MASS_MAX, size=nsamples)
    q_array = np.random.uniform(Q_MIN, Q_MAX, size=nsamples)

    m1_array = Mc_array * ((1 + q_array)**(1/5) / q_array**(3/5))
    m2_array = m1_array * q_array

    #z_array = np.random.uniform(Z_MIN, Z_MAX, size=nsamples)
    #d_array = Planck18.luminosity_distance(z_array).value

    d_array = 10**np.random.uniform(3, 5, size=nsamples)

    spin1_array = np.random.uniform(SPIN_MIN, SPIN_MAX, size=nsamples)
    spin2_array = np.random.uniform(SPIN_MIN, SPIN_MAX, size=nsamples)
    iota_array = np.arccos(np.random.uniform(-1, 1, size=nsamples))

    gw_beta_array = np.arcsin(np.random.uniform(-1, 1, size=nsamples))
    gw_lambda_array = np.random.uniform(0, 2*np.pi, size=nsamples)
    t0_array = ORB_TO + np.random.uniform(0.01, 0.99, size=nsamples)*ORB_SIZE*ORB_DT

    """d_array = 1e4*np.ones(nsamples)
    spin1_array, spin2_array = np.zeros(nsamples), np.zeros(nsamples)
    iota_array = np.zeros(nsamples)
    gw_beta_array = np.zeros(nsamples)
    gw_lambda_array = np.zeros(nsamples)
    t0_array = ORB_TO + 0.5*np.ones(nsamples)*ORB_SIZE*ORB_DT"""

    return m1_array, m2_array, d_array, spin1_array, spin2_array, iota_array, gw_beta_array, gw_lambda_array, t0_array

def main():
    start = time.time()

    m1_array, m2_array, d_array, spin1_array, spin2_array, iota_array,\
        gw_beta_array, gw_lambda_array, t0_array = get_param_values(NSAMPLES)

    jobs = [(m1_array[i], m2_array[i], d_array[i], 
             spin1_array[i], spin2_array[i], iota_array[i],
             gw_beta_array[i], gw_lambda_array[i], t0_array[i], PIPE) for i in range(NSAMPLES)]

    #if os.path.exists(FILE_PATH):
     #   os.remove(FILE_PATH)
    
    h5file = create_gw_dataset(FILE_PATH, PIPE['size'])

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
            
            for tdi_dict, m1, m2, d, spin1, spin2, iota, gw_beta, gw_lambda, t0 in results:
                append_gw_sample(h5file, tdi_dict, m1, m2, d, spin1, spin2, iota, gw_beta, gw_lambda, t0)

    #sort_gw_dataset(h5file)
    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()