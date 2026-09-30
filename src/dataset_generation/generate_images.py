# Generate Q-transform image stacks for samples containing both transients.
import os
# Limit numerical-library threading so each worker uses one CPU thread.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
import time
import traceback
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

FILE_PATH = training_dataset_path

# Image-stack dimensions and parallel processing settings.
RESOLUTION = 128
TIME_STEP = 5
CHANNELS = 4
N_WORKERS, CHUNKSIZE = 6, 2

def get_centering_times(metadata):
    if metadata['smbhb'] and metadata['glitch']:
        gw_tinj = metadata['smbhb']['t_inj']
        glitch_tinj = metadata['glitch']['t_inj']
        sep = np.abs((glitch_tinj - gw_tinj)/(2*3600))
        tcen_arr = np.random.uniform(-(3-sep), 3-sep, size=TIME_STEP)*3600
        tcen0 = PIPE['size']*PIPE['dt']/2
    else:
        tcen_arr = np.random.uniform(-3, 3, size=TIME_STEP)*3600
        tcen0 = PIPE['size']*PIPE['dt']/2

    return tcen_arr, tcen0

def get_label(metadata):
    if metadata['smbhb'] and metadata['glitch']:
        label = [1, 1]
    elif metadata['smbhb'] and not metadata['glitch']:
        label = [1, 0]
    elif not metadata['smbhb'] and metadata['glitch']:
        label = [0, 1]
    else:
        label = [0, 0]

    return label

def run_one_sample(args):
    sim_idx, tdi_dict, metadata = args

    # Draw offsets in hours, then express them in seconds for the Q-scan helper.
    tcen_arr, tcen0 = get_centering_times(metadata)
    label = get_label(metadata)
    
    images = np.zeros((len(tcen_arr), RESOLUTION, RESOLUTION, CHANNELS))
    image_metadatas, labels = [], []

    for i, tcen in enumerate(tcen_arr):
        event = tcen0 - tcen
        QT = np.zeros((RESOLUTION, RESOLUTION, CHANNELS))
        t_axis = np.zeros((RESOLUTION, CHANNELS))
        f_axis = np.zeros((RESOLUTION, CHANNELS))

        # Use both A/E channels and two Q values to form four image channels.
        j = 0
        for channel in ['A', 'E']:
            for spec in [SPECS['q6'], SPECS['q16']]:
                t_arr, f_arr, data = generate_qscan(tdi_dict, channel, PIPE, event, resolution=RESOLUTION,
                                    frange=spec['frange'], trange=spec['trange'], Q=spec['Q'])

                QT[:, :, j] = data
                t_axis[:, j] = t_arr
                f_axis[:, j] = f_arr
                j += 1
        
        images[i] = QT
        labels.append(label)
        image_metadatas.append({'t_axis': t_axis, 'f_axis':f_axis,
                                'tcen':tcen, 'sim_idx':sim_idx, 'time_idx':i})

    return images, labels, image_metadatas

def run_chunk(job_chunk):
    # Process a group of samples while isolating individual failures.
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

def create_jobs(h5file):
    if len(h5file['image_metadata/sim_idx']) > 0:
        n_images = h5file['image_metadata/sim_idx'][-1] + 1
    else:
        n_images = 0
    n_signals = h5file['tdi/X'].shape[0]

    # If all signals already have images, exit early
    if n_images >= n_signals:
        return None

    # Otherwise continue from where we left off
    X_array = h5file['tdi/X'][n_images:]
    Y_array = h5file['tdi/Y'][n_images:]
    Z_array = h5file['tdi/Z'][n_images:]

    # Prepare job list
    jobs = []
    for i in range(len(X_array)):
        tdi_dict = {}
        A,E,T = get_AET(X_array[i], Y_array[i], Z_array[i])
        tdi_dict['A'] = A
        tdi_dict['E'] = E
        tdi_dict['T'] = T

        metadata = {'smbhb': json.loads(h5file['metadata/smbhb_data'][n_images + i]),
                    'glitch': json.loads(h5file['metadata/glitch_data'][n_images + i])}
        jobs.append((n_images + i, tdi_dict, metadata))

    return jobs

def main():
    # Add Q-transform stacks for every dataset signal not yet represented in the file.
    start = time.time()

    h5file = h5py.File(FILE_PATH, "r+")

    # Reuse the class HDF5 file and create image datasets only when absent.
    if 'images' not in h5file:
        create_imageset(h5file, TIME_STEP, RESOLUTION, CHANNELS)

    jobs = create_jobs(h5file)
    if jobs is None:
        print("All signals already have generated images. Nothing to do.")
        h5file.close()
        return None

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
            
            for images, labels, image_metadatas in results:
                for i in range(len(image_metadatas)):
                    append_image(h5file, images[i], labels[i], image_metadatas[i])
    
    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()