import os

import torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from posixpath import sep
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

from model_training.model import CNN, LISADataset

import torch
from torch.utils.data import random_split, DataLoader, Subset

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


MIXED_SIM_INFERENCE_PATH = INFERENCE_DATASETS / 'seperation_analysis' / 'mixed_sim_inference.h5'

RESOLUTION = 256
N_WORKERS, CHUNKSIZE = 6, 2
TIME_STEP = 20
MASS_CATEGORY = 'low_mass'  # 'low_mass', 'high_mass', 'ext_high_mass'

DEVICE = torch.device('mps')
TRANSIENT_NAMES = {1: 'gw', 2: 'glitch', 3: 'mixed'}

SAVE_DIR = INFERENCE_DATASETS / 'timecen_datasets' / MASS_CATEGORY

TRANSIENT_DICT = {1: {'low_mass': gw_lm_dataset_path, 'high_mass': gw_hm_dataset_path, 'ext_high_mass': gw_ehm_dataset_path},
                  2: {'low_mass': glitch_lm_dataset_path, 'high_mass': glitch_hm_dataset_path, 'ext_high_mass': glitch_ehm_dataset_path},
                  3: {'low_mass': mixed_lm_dataset_path, 'high_mass': mixed_hm_dataset_path, 'ext_high_mass': mixed_ehm_dataset_path}}

MODEL_PARAMS = MODEL_PARAM_DICT[MASS_CATEGORY]
SPEC = SPECS[MASS_CATEGORY]

if MASS_CATEGORY == 'low_mass':
    DATASET_PATH = training_lm_dataset_path
    WEIGHTS_PATH = model_lm_weights_path
elif MASS_CATEGORY == 'high_mass':
    DATASET_PATH = training_hm_dataset_path
    WEIGHTS_PATH = model_hm_weights_path
elif MASS_CATEGORY == 'ext_high_mass':
    DATASET_PATH = training_ehm_dataset_path
    WEIGHTS_PATH = model_ehm_weights_path

def run_one_sample(args):
    m1, m2, d, spin1, spin2, gw_beta, gw_lambda, amp, beta, inj_point, sep, t0, pipe = args

    pipe['t0'] = t0
    gws = [{'type': 'BinaryInspiralGW',
        'm1': m1, 'm2': m2, 'd': d,
        'spin1': spin1, 'spin2': spin2,
        'gw_beta': gw_beta, 'gw_lambda': gw_lambda,
        't_inj': pipe['t_inj'] + sep * 3600, 'domain': 'freq'}]

    glitches = [{'type':'IntegratedShapeletGlitch', 't_inj': pipe['t_inj'] - sep * 3600, 'beta':beta,
             'level':2*amp*beta, 'inj_point':inj_point}]

    pid = os.getpid()
    local_sim_path = f"{simulation_path}_{pid}.h5"
    local_gw_path     = f"{gw_path}_{pid}.h5"
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

    return tdi_dict, m1, m2, d, spin1, spin2,\
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

def split_dataset(dataset, train_frac=0.7, val_frac=0.2):
    unique_indices = np.unique(dataset.sim_indices)
    all_indices = np.array(dataset.sim_indices)

    rng = np.random.default_rng(42)
    rng.shuffle(unique_indices)

    n_total = len(unique_indices)
    n_train = int(train_frac * n_total)
    n_val   = int(val_frac * n_total)

    train_sources = unique_indices[:n_train]
    val_sources   = unique_indices[n_train:n_train+n_val]
    test_sources  = unique_indices[n_train+n_val:]

    train_indices = np.where(np.isin(all_indices, train_sources))[0]
    val_indices   = np.where(np.isin(all_indices, val_sources))[0]
    test_indices  = np.where(np.isin(all_indices, test_sources))[0]

    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)

def evaluate_model(model, dataset):
    loader = DataLoader(dataset, batch_size=64)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    pred_class = preds[:,0]*1 + preds[:,1]*2
    true_class = labels[:,0]*1 + labels[:,1]*2
    return pred_class.numpy(), true_class.numpy()

def get_param_values(model):
    dataset = LISADataset(DATASET_PATH)
    _, _, test_dataset = split_dataset(dataset)

    pred_class, true_class = evaluate_model(model, test_dataset)
    print('Evaluation complete...')

    correct = np.where((true_class == 3) & (pred_class == 3))[0]
    sim_indices = np.array([test_dataset[i][2] for i in correct])
    sim_indices = np.unique(sim_indices)

def main():
    start = time.time()

    m1_array, m2_array, d_array, spin1_array, spin2_array, gw_beta_array, gw_lambda_array,\
         amp_array, beta_array, inj_point_array, sep_array, t0_array = get_param_values()

    jobs = [(m1_array[i], m2_array[i], d_array[i], 
             spin1_array[i], spin2_array[i], 
             gw_beta_array[i], gw_lambda_array[i],
             amp_array[i], beta_array[i], inj_point_array[i],
             sep_array[i], t0_array[i], PIPE) for i in range(len(m1_array))]

    if os.path.exists(MIXED_SIM_INFERENCE_PATH):
        os.remove(MIXED_SIM_INFERENCE_PATH)

    h5file = create_mixed_dataset(MIXED_SIM_INFERENCE_PATH, PIPE['size'])

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
            
            for tdi_dict, m1, m2, d, spin1, spin2, gw_beta, gw_lambda, amp, beta, inj_point, sep, t0 in results:
                append_mixed_sample(h5file, tdi_dict, m1, m2, d, spin1, spin2, gw_beta, gw_lambda,
                                    amp, beta, inj_point, sep, t0)

    sort_mixed_dataset(h5file)
    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()