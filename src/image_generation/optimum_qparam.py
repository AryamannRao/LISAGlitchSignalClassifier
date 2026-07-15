# Keep these import statements and add anything else that might be relevant
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import time

# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.qtransform import *
from helpers.config import *
from helpers.simulation import *

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import h5py

SAVE_PATH = GW_DATASETS / 'optimum_q.npz'
N_WORKERS, CHUNKSIZE = 6, 2
Q_VALS = np.array([4, 8, 16])

def run_one_sample(args):
    index, tdi_dict = args
    event = 5*3600
    scores = []

    for Q in Q_VALS:
        t_arr, f_arr, QT = generate_qscan(tdi_dict, 'A', PIPE, event, resolution=256,
                                frange=(1e-4, 1e-1), trange=(-3, 3), Q=Q)
        
        QTnorm = QT/np.max(QT)
        scores.append(len(QTnorm[QTnorm > 0.8]))

        #snr = np.sqrt(QT**2 - 1)

        #threshold = 3
        #area = len(snr[snr > threshold])
        #vals = np.mean(snr[snr > threshold])
        #scores.append(area*vals)
        #scores.append(np.percentile(QT, 99.9))
        #scores.append(np.max(QT))

    scores = np.array(scores) 
    Qopt = Q_VALS[scores == np.max(scores)][0]
    return index, Qopt

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

    gw_dict = {}
    with h5py.File(gw_dataset_path, 'r') as h5:
        for key in list(h5.keys()):
            gw_dict[key] = h5[key][:]
  
    jobs = []
    for i in range(gw_dict['X'].shape[0]):
        tdi_dict = make_tdi_dict(gw_dict['X'][i],\
                            gw_dict['Y'][i], gw_dict['Z'][i], whiten=True)
        jobs.append((i,tdi_dict))

    print(f"Running with {N_WORKERS} workers")

    job_chunks = list(chunkify(jobs, CHUNKSIZE))

    total_chunks = len(job_chunks)
    total_samples = len(jobs)

    completed_samples = 0
    index_array, Qopt_array = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(run_chunk, chunk) for chunk in job_chunks]

        for future in tqdm(futures, total=total_chunks, desc="Chunks completed"):
            results = future.result()
            
            completed_samples += len(results)
            tqdm.write(f"Samples done: {completed_samples}/{total_samples}")
            
            for index, Qopt in results:
                index_array.append(index)
                Qopt_array.append(Qopt)

    index_array = np.array(index_array)
    Qopt_array = np.array(Qopt_array)

    np.savez(SAVE_PATH, index=index_array, Qopt=Qopt_array)
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()