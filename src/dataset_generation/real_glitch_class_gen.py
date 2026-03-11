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
from helpers.glitch_shapes import RealGlitch

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

INJ_POINTS = ['tm_12', 'tm_23', 'tm_13',
              'tm_21', 'tm_32', 'tm_31']

N_WORKERS, CHUNKSIZE = 6, 2

def run_one_sample(args):
    run, index, inj_point, pipe = args

    gws = [{'type':'ReducedOneSidedDoubleExpGW', 't_inj': 0, 'amp':0, 't_rise':1, 't_fall':1,
             'gw_beta':0, 'gw_lambda':0}]

    glitches = [{'type':'RealGlitch', 't_inj': pipe['t_inj'], 'index':index, 'run':run,
             'scale':1e5, 'inj_point':inj_point}]
    
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

    return tdi_dict, run, index, inj_point

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

def get_param_values():
    path = str(DIST / 'lpf-glitch-library.h5')
    h5 = h5py.File(path, "r")

    run_no = 1
    glitch_no = 1
    count = 0

    run_array, index_array = [], []
    while run_no <= 76:
        key = f'timeseries/run{run_no:02d}/glitch{glitch_no:02d}'
        if key in h5:
            count += 1
            index_array.append(glitch_no)
            run_array.append(run_no)
        glitch_no += 1
        
        if key not in h5:
            run_no += 1
            glitch_no = 1

    h5.close()
    inj_point_array = np.random.choice(INJ_POINTS, count)
    return run_array, index_array, inj_point_array

def main():
    start = time.time()

    run_array, index_array, inj_point_array = get_param_values()

    jobs = [(run_array[i], index_array[i], inj_point_array[i], PIPE) for i in range(len(run_array))]
    
    # Create dataset
    #if os.path.exists(glitch_dataset_path):
     #   os.remove(glitch_dataset_path)

    h5file = create_realglitch_dataset(real_glitch_dataset_path, PIPE['size'])

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

            for tdi_dict, run, index, inj_point in results:
                append_realglitch_sample(h5file, tdi_dict, run, index, inj_point)
    
    #sort_glitch_dataset(glitch_dataset_path)
    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
