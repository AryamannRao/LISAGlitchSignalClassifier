import os
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch.utils.data import random_split, DataLoader, Subset

from model import CNN, LISADataset
from helpers.config import *

MASS_CATEGORY = 'high_mass'  # 'low_mass', 'high_mass', 'ext_high_mass'

if MASS_CATEGORY == 'low_mass':
    DATASET_PATH = training_lm_dataset_path
elif MASS_CATEGORY == 'high_mass':
    DATASET_PATH = training_hm_dataset_path
elif MASS_CATEGORY == 'ext_high_mass':
    DATASET_PATH = training_ehm_dataset_path

SAVE_DIR = TRAINING_RESULTS / MASS_CATEGORY /f'run_{datetime.now().strftime("%d%m%y_%H%M%S")}'

MODEL_PARAMS = {'ext_high_mass':{'width': 6, 'bn': True, 'normalise': False,
                                'drop': 0.0, 'num_epochs': 9, 'learning_rate': 0.001,
                                'batch_size': 64, 'plot_every': 50},
                'high_mass':{'width': 6, 'bn': True, 'normalise': False,
                            'drop': 0.0, 'num_epochs': 6, 'learning_rate': 0.001,
                            'batch_size': 64, 'plot_every': 33},
                'low_mass':{'width': 6, 'bn': True, 'normalise': False,
                            'drop': 0.0, 'num_epochs': 9, 'learning_rate': 0.001,
                            'batch_size': 64, 'plot_every': 50}}[MASS_CATEGORY]

WIDTH = MODEL_PARAMS['width']
USE_BATCH_NORM = MODEL_PARAMS['bn']
NORMALISE = MODEL_PARAMS['normalise']
BATCH_SIZE = MODEL_PARAMS['batch_size']
LEARNING_RATE = MODEL_PARAMS['learning_rate']
NUM_EPOCHS = MODEL_PARAMS['num_epochs']
DROP = MODEL_PARAMS['drop']
PLOT_EVERY = MODEL_PARAMS['plot_every']

DEVICE = torch.device('mps')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

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

def accuracy(model, loader, threshold=0.5):
    model.eval()
    
    soft_correct, hard_correct, soft_total, hard_total = 0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > threshold).float()
            soft_correct += (predicted == labels).sum().item()
            hard_correct += ((predicted == labels).all(dim=1)).sum().item()
            soft_total += labels.numel()
            hard_total += labels.size(0)
    
    soft_acc = soft_correct/soft_total
    hard_acc = hard_correct/hard_total
    return soft_acc, hard_acc

def compute_loss(model, loader, criterion):
    model.eval()
    loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(images)
            loss += float(criterion(outputs, labels))
            count += 1
    loss /= count
    return loss

def train_model(model, train_data, val_data, test_data,
                learning_rate=0.005, batch_size=10,
                num_epochs=10, plot_every=10):
    model = model.to(DEVICE)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    subset_size = 500
    rng = np.random.default_rng(42)
    indices = rng.choice(len(val_data), subset_size, replace=False)

    val_subset = Subset(val_data, indices)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    iters, train_loss, valid_loss = [], [], []
    iter_count = 0
    best_loss = np.inf

    start = time.time()
    for e in range(num_epochs):
        model.train()
        for batch in train_loader:
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            z = model(images)
            loss = criterion(z, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if float(loss.detach()) < best_loss:
                    best_loss = float(loss.detach())
                    torch.save(model.state_dict(), SAVE_DIR / 'best_model_weights.pth')
                    print(f'Best model weights saved at iteration {iter_count} and loss = {float(loss.detach()):.3f}')

            if iter_count % plot_every == 0:
                iters.append(iter_count)
                train_loss.append(float(loss.detach()))
                valid_loss.append(compute_loss(model, val_loader, criterion))
                model.train()
                
                end = time.time()
                print(f"Epoch {e+1}/{num_epochs}, Iteration {iter_count}, Loss: {loss:.3f}, " +\
                        f"Val loss: {valid_loss[-1]:.3f}, " +\
                        f"Time passed: {(end - start):.3f} secs.")
            iter_count += 1
    
    iters.append(iter_count)
    train_loss.append(float(loss.detach()))

    print('Computing accuracies:')
    val_loader = DataLoader(val_data, batch_size=batch_size)
    val_soft_acc, val_hard_acc = accuracy(model, val_loader)

    test_loader = DataLoader(test_data, batch_size=batch_size)
    test_soft_acc, test_hard_acc = accuracy(model, test_loader)

    return iters, train_loss, valid_loss,\
         val_soft_acc, val_hard_acc, test_soft_acc, test_hard_acc

def plot_results(results):
    iters, train_loss, valid_loss,\
         val_soft_acc, val_hard_acc, test_soft_acc, test_hard_acc = results

    fig, axes = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)
    title = f"Model used: CNN (width={WIDTH}, batch norm={USE_BATCH_NORM}, norm={NORMALISE}, drop={DROP})\n" +\
    f"Training curve (batch size={BATCH_SIZE}, learning rate={LEARNING_RATE}, num epochs={NUM_EPOCHS})\n" +\
    f"Final Valid Soft Acc: {val_soft_acc:.3f}, Final Valid Hard Acc: {val_hard_acc:.3f},\n" +\
    f"Test Soft Acc: {test_soft_acc:.3f}, Test Hard Acc: {test_hard_acc:.3f}, \n" +\
    f"Final Train loss {train_loss[-1]:.3f}, Final Val loss {valid_loss[-1]:.3f}"
    fig.suptitle(title, fontsize=12)

    axes.plot(iters[:len(train_loss)], train_loss, label='Train loss')
    axes.plot(iters[:len(valid_loss)], valid_loss, label='Validation loss')
    axes.set_title("Loss over iterations")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Loss")
    axes.legend()
    plt.savefig(SAVE_DIR / 'training_curves.png')

def main():
    set_seed(42)

    model = CNN(width=WIDTH, bn=USE_BATCH_NORM, normalise=NORMALISE, drop=DROP)
    dataset = LISADataset(DATASET_PATH)
    if not os.path.exists(SAVE_DIR):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    
    results = train_model(model, train_dataset, val_dataset, test_dataset,
                        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, 
                        num_epochs=NUM_EPOCHS, plot_every=PLOT_EVERY)
    
    torch.save(model.state_dict(), SAVE_DIR / 'model_weights.pth')
    torch.save(results, SAVE_DIR / 'results.pt')
    plot_results(results)

if __name__ == "__main__":
    main()