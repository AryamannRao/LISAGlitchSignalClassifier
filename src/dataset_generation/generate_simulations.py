# Generate binary-inspiral gravitational-wave samples for the GW dataset class.
import os
# Limit numerical-library threading so each worker uses one CPU thread.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import contextlib
import numpy as np
import time
import traceback
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

N_SMBHB_MERG, N_GLITCH = 0, 0
NSAMPLES = 1500
TEST = False
FILE_PATH = str(TRAINING_DATASETS / f'dataset_{N_SMBHB_MERG}{N_GLITCH}.h5')

 # Bounds used when sampling source masses, distances, spins, and orientations.
LOG_MASS_MIN, LOG_MASS_MAX = 4, 7
LOG_D_MIN, LOG_D_MAX = 3, 5
Q_MIN, Q_MAX = 1, 5
SPIN_MIN, SPIN_MAX = -0.99, 0.99

# Define the glitch-parameter sampling ranges in log space.
LOG_BETA_MIN, LOG_BETA_MAX = 0, 4
LOG_AMP_MIN, LOG_AMP_MAX = -14, -11
INJ_POINTS = ['tm_12', 'tm_23', 'tm_13',
              'tm_21', 'tm_32', 'tm_31']

SEP_MIN, SEP_MAX = 0, 3

N_WORKERS, CHUNKSIZE = 6, 2

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

def get_smbhb_params(nsamples):
    # Sample physically distributed source parameters and valid orbit start times.
    if TEST:
        Mc_array = 10**np.random.uniform(LOG_MASS_MIN, LOG_MASS_MAX, size=nsamples)
        d_array = 10**np.random.uniform(LOG_D_MIN, LOG_D_MAX, size=nsamples)

    else:
        smbhb_mass_params = get_filtered_params(nsamples, smbhb_forest_path, 
                                              lower=[LOG_D_MIN, LOG_MASS_MIN],
                                              upper=[LOG_D_MAX, LOG_MASS_MAX])
        Mc_array = 10**smbhb_mass_params[:,1]
        d_array = 10**smbhb_mass_params[:,0] 

    q_array = np.random.uniform(Q_MIN, Q_MAX, size=nsamples)

    # Convert chirp mass and mass ratio into component masses.
    m1_array = Mc_array * ((1 + q_array)**(1/5) / q_array**(3/5))
    m2_array = m1_array * q_array

    spin1_array = np.random.uniform(SPIN_MIN, SPIN_MAX, size=nsamples)
    spin2_array = np.random.uniform(SPIN_MIN, SPIN_MAX, size=nsamples)
    iota_array = np.arccos(np.random.uniform(-1, 1, size=nsamples))

    gw_beta_array = np.arcsin(np.random.uniform(-1, 1, size=nsamples))
    gw_lambda_array = np.random.uniform(0, 2*np.pi, size=nsamples)

    return {'m1':m1_array, 'm2':m2_array, 'd':d_array, 'spin1':spin1_array, 'spin2':spin2_array,
               'iota': iota_array, 'gw_beta':gw_beta_array, 'gw_lambda': gw_lambda_array}

def get_glitch_params(nsamples):
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

    return {'level':2*amp_array*beta_array, 'beta':beta_array, 'inj_point':inj_point_array}

def get_inj_times(smbhb_mergers, glitches):
    if len(smbhb_mergers) > 0 and len(glitches) > 0:
        sep = np.random.choice([-1, 1])*np.random.uniform(SEP_MIN, SEP_MAX)
        smbhb_mergers[0]['t_inj'] = PIPE['size']*PIPE['dt']/2 - sep * 3600
        glitches[0]['t_inj'] = PIPE['size']*PIPE['dt']/2 + sep * 3600

    elif len(smbhb_mergers) > 0:
        smbhb_mergers[0]['t_inj'] = PIPE['size']*PIPE['dt']/2
    elif len(glitches) > 0:
        glitches[0]['t_inj'] = PIPE['size']*PIPE['dt']/2

    return smbhb_mergers, glitches

def create_jobs(samples, t0_array):
    jobs = []
    for i in range(NSAMPLES):
        smbhb_mergers, glitches = [], []
        pipe = PIPE.copy()
        pipe['t0'] = t0_array[i]

        if 'smbhb_merger' in samples.keys():
            smbhb_dict = {}
            for key in samples['smbhb_merger'].keys():
                smbhb_dict[key] = samples['smbhb_merger'][key][i]
            smbhb_dict['type'] = 'SMBHB'
            smbhb_dict['domain'] = 'freq'
            smbhb_mergers.append(smbhb_dict)

        if 'glitch' in samples.keys():
            glitch_dict = {}
            for key in samples['glitch'].keys():
                glitch_dict[key] = samples['glitch'][key][i]
            glitch_dict['type'] = 'IntegratedShapeletGlitch'
            glitches.append(glitch_dict)

        smbhb_mergers, glitches = get_inj_times(smbhb_mergers, glitches)

        jobs.append((smbhb_mergers, glitches, pipe))

    return jobs

def run_one_sample(args):
    # Construct, simulate, and transform one binary-inspiral injection.
    smbhbs, glitches, pipe = args

    pid = os.getpid()
    local_sim_path = f"{simulation_path}_{pid}.h5"
    local_smbhb_path = f"{smbhb_path}_{pid}.h5" if len(smbhbs) > 0 else None
    local_glitch_path = f"{glitch_path}_{pid}.h5" if len(glitches) > 0 else None

    with suppress_output():
        run_simulation(
            smbhbs, glitches, pipe,
            local_smbhb_path, local_glitch_path, orbits_path,
            local_sim_path,
            disable_noise=pipe['keep_noises'])
        
        tdi_dict = run_tdi(local_sim_path, pipe)

    os.remove(local_sim_path)

    if local_smbhb_path:
        os.remove(local_smbhb_path)

    if local_glitch_path:
        os.remove(local_glitch_path)

    metadata = {'smbhb_data': smbhbs[0] if smbhbs else None,
                'glitch_data': glitches[0] if glitches else None,
                't0': pipe['t0']}
    return tdi_dict, metadata

def run_chunk(job_chunk):
    # Keep failures local to individual jobs within a worker chunk.
    results = []
    for job in job_chunk:
        try:
            results.append(run_one_sample(job))
        except Exception as e:
            print(f"Error processing job: {e}")
            traceback.print_exc()
            continue
    return results

def chunkify(lst, chunksize):
    # Yield fixed-size job groups for submission to the process pool.
    for i in range(0, len(lst), chunksize):
        yield lst[i:i + chunksize]

def main():
    # Create the output dataset and populate it with parallel simulations.
    start = time.time()

    transients = {'smbhb_merger': {'num': N_SMBHB_MERG, 'sampler': get_smbhb_params},
                  'glitch': {'num': N_GLITCH, 'sampler': get_glitch_params}}

    samples = {}
    for transient in transients.keys():
        if transients[transient]['num'] > 0:
            samples[transient] = transients[transient]['sampler'](NSAMPLES)

    t0_array = ORB_TO + np.random.uniform(0.01, 0.99, size=NSAMPLES)*ORB_SIZE*ORB_DT

    jobs = create_jobs(samples, t0_array)

    h5file = create_simulation_dataset(FILE_PATH)
    
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
            
            for tdi_dict, metadata in results:
                append_sample(h5file, tdi_dict, metadata)

    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()




