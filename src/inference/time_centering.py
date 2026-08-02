import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
import time
from pathlib import Path

# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.simulation import *
from helpers.h5file_helpers import *
from helpers.qtransform import *
from helpers.config import *

from model_training.model import CNN, LISADataset

import torch
from torch.utils.data import random_split, DataLoader, Subset

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

RESOLUTION, CHANNELS = 256, 2
N_WORKERS, CHUNKSIZE = 6, 2
TIME_STEP = 25

DEVICE = torch.device('mps')
TRANSIENT_SIMPATHS = {1: gw_dataset_path, 2: glitch_dataset_path}
TRANSIENT_RESULTPATHS = {1: gw_timecen_path, 2: glitch_timecen_path}

RESULTS_DIR = TRAINING_RESULTS / 'run_010826_203219'
DATASET_PATH = training_dataset_path
WEIGHTS_PATH = RESULTS_DIR / 'model_weights.pth'

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

    return train_dataset, val_dataset, test_dataset

def load_best_model(weights_path, model_params):
    model = CNN(width=model_params['width'], normalise=model_params['normalise'],\
                 bn=model_params['bn'], drop=model_params['drop'])
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

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

def run_one_sample(args):
    X, Y, Z = args
    tdi_dict = make_tdi_dict(X, Y, Z, whiten=False)
    tcen_arr = np.linspace(-2.5, 2.5, TIME_STEP)*3600
    
    images = np.zeros((len(tcen_arr), RESOLUTION, RESOLUTION, CHANNELS))
    t_axes = np.zeros((len(tcen_arr), RESOLUTION, CHANNELS))
    f_axes = np.zeros((len(tcen_arr), RESOLUTION, CHANNELS))
    tcen_vals = np.zeros(len(tcen_arr))

    for i, tcen in enumerate(tcen_arr):
        event = PIPE['t_inj'] - tcen
        QT = np.zeros((RESOLUTION, RESOLUTION, CHANNELS))
        t_axis = np.zeros((RESOLUTION, CHANNELS))
        f_axis = np.zeros((RESOLUTION, CHANNELS))
        for j, spec in enumerate([SPECS['q6'], SPECS['q16']]):
            t_arr, f_arr, data = generate_qscan(tdi_dict, 'A', PIPE, event, resolution=RESOLUTION,
                                frange=spec['frange'], trange=spec['trange'], Q=spec['Q'])

            QT[:, :, j] = data
            t_axis[:, j] = t_arr
            f_axis[:, j] = f_arr
        
        images[i], t_axes[i], f_axes[i], tcen_vals[i] = QT, t_axis, f_axis, tcen

    return images, t_axes, f_axes, tcen_vals

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

def create_inference_set(model, transient_type, keys=['QT', 't_qscan', 'f_qscan', 'tcen']):
    dataset = LISADataset(DATASET_PATH)
    _, _, test_dataset = split_dataset(dataset)

    pred_class, true_class = evaluate_model(model, test_dataset)
    print('Evaluation complete...')

    correct = np.where((true_class == transient_type) & (pred_class == transient_type))[0]
    sim_indices = np.array([test_dataset[i][2] for i in correct])
    sim_indices = np.unique(sim_indices)
    print(f"Found {len(sim_indices)} correctly classified samples for chosen transient type.")

    simulation_path = TRANSIENT_SIMPATHS[transient_type]
    with h5py.File(simulation_path, 'r') as f:
        X_array = f['X'][sim_indices]
        Y_array = f['Y'][sim_indices]
        Z_array = f['Z'][sim_indices]

    # Prepare job list
    jobs = [(X_array[i], Y_array[i], Z_array[i]) for i in range(len(X_array))]

    print(f"Running with {N_WORKERS} workers")

    job_chunks = list(chunkify(jobs, CHUNKSIZE))

    total_chunks = len(job_chunks)
    total_samples = len(jobs)

    completed_samples = 0
    keys = ['QT', 't_qscan', 'f_qscan']

    save_path = TRANSIENT_RESULTPATHS[transient_type]
    h5file = create_imageset(save_path, TIME_STEP, RESOLUTION, CHANNELS, keys=keys)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(run_chunk, chunk) for chunk in job_chunks]

        for future in tqdm(futures, total=total_chunks, desc="Chunks completed"):
            results = future.result()
            
            completed_samples += len(results)
            tqdm.write(f"Samples done: {completed_samples}/{total_samples}")
            
            for images, t_axes, f_axes, tcen in results:
                append_image(h5file, images, t_axes, f_axes, tcen, keys=keys)
    
    h5file.create_dataset('sim_indices', data=sim_indices)
    h5file.close()

def run_inference(model):
    with h5py.File(gw_timecen_path, 'r') as f:
        gws = f['QT'][:]
    with h5py.File(glitch_timecen_path, 'r') as f:
        glitches = f['QT'][:]
        tcen_vals = f['tcen'][:]

    gw_acc, glitch_acc, gw_std, glitch_std = [], [], [], []
    with torch.no_grad():
        for i in range(TIME_STEP):
            gw_images = torch.tensor(gws[:, i, :, :, :],\
                                    dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
            glitch_images = torch.tensor(glitches[:, i, :, :, :],\
                                        dtype=torch.float32).permute(0, 3, 1, 2).contiguous()

            gw_outputs = model(gw_images.to(DEVICE))
            glitch_outputs = model(glitch_images.to(DEVICE))

            gw_probs = torch.sigmoid(gw_outputs)
            glitch_probs = torch.sigmoid(glitch_outputs)

            gw_acc.append(np.mean(gw_probs.to('cpu').numpy(), axis=0))
            glitch_acc.append(np.mean(glitch_probs.to('cpu').numpy(), axis=0))

            gw_std.append(np.std(gw_probs.to('cpu').numpy(), axis=0))
            glitch_std.append(np.std(glitch_probs.to('cpu').numpy(), axis=0))

    return tcen_vals[0,:], gw_acc, glitch_acc, gw_std, glitch_std

def main():
    model = load_best_model(WEIGHTS_PATH, MODEL_PARAMS)

    if not os.path.exists(gw_timecen_path):
        print("Creating GW inference dataset...")
        create_inference_set(model, transient_type=1)
    
    if not os.path.exists(glitch_timecen_path):
        print("Creating Glitch inference dataset...")
        create_inference_set(model, transient_type=2)
    
    if os.path.exists(gw_timecen_path) and os.path.exists(glitch_timecen_path):
        print("Running inference...")
        times, gw_acc, glitch_acc, gw_std, glitch_std = run_inference(model)
        np.savez(timecen_result_path, times=times, gw_acc=gw_acc,\
                  glitch_acc=glitch_acc, gw_std=gw_std, glitch_std=glitch_std)


if __name__ == "__main__":
    main()