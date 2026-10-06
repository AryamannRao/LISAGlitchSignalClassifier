# Evaluate trained classifications as a function of Q-scan time centring.
import os
# Limit numerical-library threading so each worker uses one CPU thread.
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

RESOLUTION, CHANNELS = 128, 4
# Inference image-stack dimensions and parallel processing settings.
N_WORKERS, CHUNKSIZE = 6, 2
TIME_STEP = 25

DEVICE = torch.device('mps')
# Map the model's binary-label class values to source and result files.
smbhb_dataset_path = str(TRAINING_DATASETS/'dataset_10.h5')
glitch_dataset_path = str(TRAINING_DATASETS/'dataset_01.h5')

TRANSIENT_SIMPATHS = {1: smbhb_dataset_path, 2: glitch_dataset_path}
TRANSIENT_RESULTPATHS = {1: smbhb_timecen_path, 2: glitch_timecen_path}

RESULTS_DIR = TRAINING_RESULTS / 'run_041026_181522'
DATASET_PATH = training_dataset_path
WEIGHTS_PATH = RESULTS_DIR / 'best_model_weights.pth'

def split_dataset(dataset, train_frac=0.7, val_frac=0.2):
    sim_indices = np.asarray(dataset.sim_indices)
    labels = np.asarray(dataset.labels)

    # Each (label, sim_idx) pair identifies one underlying simulation.
    groups = np.unique(np.column_stack((labels, sim_indices)),axis=0)

    rng = np.random.default_rng(42)
    rng.shuffle(groups)

    n_total = len(groups)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train+n_val]
    test_groups = groups[n_train+n_val:]

    def get_indices(group_set):
        mask = np.zeros(len(dataset), dtype=bool)

        for label0, label1, sim_idx in group_set:
            mask |= ((labels[:, 0] == label0) &
                (labels[:, 1] == label1) &
                (sim_indices == sim_idx))

        return np.where(mask)[0]

    train_indices = get_indices(train_groups)
    val_indices = get_indices(val_groups)
    test_indices = get_indices(test_groups)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def load_best_model(weights_path, model_params):
    # Recreate the configured CNN and load its saved best checkpoint for inference.
    model = CNN(width=model_params['width'], normalise=model_params['normalise'],\
                 bn=model_params['bn'], drop=model_params['drop'])
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def evaluate_model(model, dataset):
    # Return predicted and true binary-encoded class values for a dataset subset.
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

    # Encode two independent classifier outputs as smbhb=1, glitch=2, mixed=3.
    pred_class = preds[:,0]*1 + preds[:,1]*2
    true_class = labels[:,0]*1 + labels[:,1]*2
    return pred_class.numpy(), true_class.numpy()

def get_centering_times():
    tcen_arr = np.linspace(-3, 3, num=TIME_STEP)*3600
    tcen0 = PIPE['size']*PIPE['dt']/2

    return tcen_arr, tcen0

def get_label(metadata):
    if metadata['smbhb'] and not metadata['glitch']:
        label = [1, 0]
    elif not metadata['smbhb'] and metadata['glitch']:
        label = [0, 1]

    return label

def create_jobs(h5file, start, batch_size):
    n_signals = h5file['tdi/X'].shape[0]

    end = min(start + batch_size, n_signals)

    if start >= n_signals:
        return None

    X_array = h5file['tdi/X'][start:end]
    Y_array = h5file['tdi/Y'][start:end]
    Z_array = h5file['tdi/Z'][start:end]

    jobs = []

    for i in range(end - start):
        sim_idx = start + i

        A, E, T = get_AET(X_array[i],Y_array[i], Z_array[i])
        tdi_dict = {'A': A, 'E': E, 'T': T}

        metadata = {
            'smbhb': json.loads(
                h5file['metadata/smbhb_data'][sim_idx]),
            'glitch': json.loads(
                h5file['metadata/glitch_data'][sim_idx])
        }

        jobs.append((sim_idx, tdi_dict, metadata))

    return jobs

def run_one_sample(args):
    sim_idx, tdi_dict, metadata = args

    # Draw offsets in hours, then express them in seconds for the Q-scan helper.
    tcen_arr, tcen0 = get_centering_times()
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
    # Process a group of simulations while isolating individual failures.
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

def create_inference_set(model, transient_type):
    # Create Q-scan sequences from correctly classified test simulations of one class.
    dataset = LISADataset(DATASET_PATH)
    _, _, test_dataset = split_dataset(dataset)

    pred_class, true_class = evaluate_model(model, test_dataset)
    print('Evaluation complete...')

    # Keep only examples whose predicted class matches the requested true class.
    correct = np.where((true_class == transient_type) & (pred_class == transient_type))[0]
    sim_indices = np.array([test_dataset[i][2] for i in correct])
    sim_indices = np.unique(sim_indices)
    print(f"Found {len(sim_indices)} correctly classified samples for chosen transient type.")

    simulation_path = TRANSIENT_SIMPATHS[transient_type]
    h5 = h5py.File(simulation_path, 'r')
    jobs = []

    for sim_idx in sim_indices:
        jobs.extend(create_jobs(h5,start=int(sim_idx),batch_size=1))
    h5.close()

    print(f"Running with {N_WORKERS} workers")

    job_chunks = list(chunkify(jobs, CHUNKSIZE))

    total_chunks = len(job_chunks)
    total_samples = len(jobs)

    completed_samples = 0

    save_path = TRANSIENT_RESULTPATHS[transient_type]
    h5file = h5py.File(save_path, 'w')
    h5file = create_imageset(h5file, RESOLUTION, CHANNELS)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(run_chunk, chunk)
                for chunk in job_chunks]

            for future in tqdm(as_completed(futures),
                total=total_chunks,desc="Chunks completed"):

                results = future.result()
                completed_samples += len(results)

                tqdm.write(f"Samples done: "f"{completed_samples}/{total_samples}")

                for images, labels, image_metadatas in results:
                    for i in range(len(image_metadatas)):
                        append_image(h5file, images[i], labels[i], image_metadatas[i])

    h5file.close()

def run_inference(model):
    with h5py.File(smbhb_timecen_path, 'r') as f:
        smbhbs = f['images'][:]
        smbhb_time_idx = f['image_metadata/time_idx'][:]
        smbhb_tcen = f['image_metadata/tcen'][:]

    with h5py.File(glitch_timecen_path, 'r') as f:
        glitches = f['images'][:]
        glitch_time_idx = f['image_metadata/time_idx'][:]

    smbhb_acc = []
    glitch_acc = []
    smbhb_std = []
    glitch_std = []

    with torch.no_grad():
        for i in range(TIME_STEP):
            # All simulations' images at the same time-centering.
            smbhb_indices = np.where(smbhb_time_idx == i)[0]
            glitch_indices = np.where(glitch_time_idx == i)[0]

            smbhb_images = torch.tensor(
                smbhbs[smbhb_indices],
                dtype=torch.float32
            ).permute(0, 3, 1, 2).contiguous()

            glitch_images = torch.tensor(
                glitches[glitch_indices],
                dtype=torch.float32
            ).permute(0, 3, 1, 2).contiguous()

            # CNN outputs.
            smbhb_probs = torch.sigmoid(
                model(smbhb_images.to(DEVICE))
            ).cpu().numpy()

            glitch_probs = torch.sigmoid(
                model(glitch_images.to(DEVICE))
            ).cpu().numpy()

            # Average over all transients having this tcen.
            smbhb_acc.append(np.mean(smbhb_probs, axis=0))
            glitch_acc.append(np.mean(glitch_probs, axis=0))

            smbhb_std.append(np.std(smbhb_probs, axis=0))
            glitch_std.append(np.std(glitch_probs, axis=0))

    times = np.array([smbhb_tcen[np.where(smbhb_time_idx == i)[0][0]]
        for i in range(TIME_STEP)])

    return (times, np.array(smbhb_acc), np.array(glitch_acc),
        np.array(smbhb_std), np.array(glitch_std))

def main():
    # Build missing inference image sets, then save time-dependent model responses.
    model = load_best_model(WEIGHTS_PATH, MODEL_PARAMS)

    if not os.path.exists(smbhb_timecen_path):
        print("Creating smbhb inference dataset...")
        create_inference_set(model, transient_type=1)
    
    if not os.path.exists(glitch_timecen_path):
        print("Creating Glitch inference dataset...")
        create_inference_set(model, transient_type=2)
    
    if os.path.exists(smbhb_timecen_path) and os.path.exists(glitch_timecen_path):
        print("Running inference...")
        times, smbhb_acc, glitch_acc, smbhb_std, glitch_std = run_inference(model)
        np.savez(timecen_result_path, times=times, smbhb_acc=smbhb_acc,\
                  glitch_acc=glitch_acc, smbhb_std=smbhb_std, glitch_std=glitch_std)

if __name__ == "__main__":
    main()
