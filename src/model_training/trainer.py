import os
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch.utils.data import random_split, DataLoader

from model import CNN, LISADataset
from helpers.config import *

MASS_CATEGORY = 'ext_high_mass'  # 'low_mass', 'high_mass', 'ext_high_mass'

if MASS_CATEGORY == 'low_mass':
    DATASET_PATH = training_lm_dataset_path
elif MASS_CATEGORY == 'high_mass':
    DATASET_PATH = training_hm_dataset_path
elif MASS_CATEGORY == 'ext_high_mass':
    DATASET_PATH = training_ehm_dataset_path

SAVE_DIR = TRAINING_RESULTS / MASS_CATEGORY /f'run_{datetime.now().strftime("%d%m%y_%H%M%S")}'

WIDTH = 6
USE_BATCH_NORM = True
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 9

PLOT_EVERY = 2
PRINT_EVERY = 15

DEVICE = torch.device('mps')

def soft_accuracy(model, dataset, threshold=0.5):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > threshold).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()
    
    return correct / total

def hard_accuracy(model, dataset, threshold=0.5):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > threshold).float()
            correct += ((predicted == labels).all(dim=1)).sum().item()
            total += labels.size(0)
    
    return correct / total

def train_model(model, train_data, val_data, test_dataset,
                learning_rate=0.005, batch_size=10,
                num_epochs=10, plot_every=10, print_every=10):
    model = model.to(DEVICE)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    iters, train_loss, train_soft_acc, val_soft_acc, train_hard_acc, val_hard_acc = [], [], [], [], [], []
    iter_count = 0
    best_loss = np.inf

    start = time.time()
    for e in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            z = model(images)
            loss = criterion(z, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if float(loss) < best_loss:
                best_loss = float(loss)
                torch.save(model.state_dict(), SAVE_DIR / 'best_model_weights.pth')
                print(f'Best model weights saved at iteration {iter_count} and loss = {float(loss):.3f}')

            iter_count += 1
            if iter_count % plot_every == 0:
                iters.append(iter_count)
                train_loss.append(float(loss))
                train_soft_acc.append(soft_accuracy(model, train_data))
                val_soft_acc.append(soft_accuracy(model, val_data))
                train_hard_acc.append(hard_accuracy(model, train_data))
                val_hard_acc.append(hard_accuracy(model, val_data))
                model.train()
                
            if iter_count % print_every == 0:
                end = time.time()
                print(f"Epoch {e+1}/{num_epochs}, Iteration {iter_count}, Loss: {loss:.3f}, " +\
                        f" Train Soft Acc: {train_soft_acc[-1]:.3f}, Val Soft Acc: {val_soft_acc[-1]:.3f}, " +\
                        f"Train Hard Acc: {train_hard_acc[-1]:.3f}, Val Hard Acc: {val_hard_acc[-1]:.3f}," +\
                        f"Time passed: {(end - start):.3f} secs.")
    
    iters.append(iter_count)
    train_loss.append(float(loss))
    train_soft_acc.append(soft_accuracy(model, train_data))
    val_soft_acc.append(soft_accuracy(model, val_data))
    train_hard_acc.append(hard_accuracy(model, train_data))
    val_hard_acc.append(hard_accuracy(model, val_data))

    test_soft_acc = soft_accuracy(model, test_dataset)
    test_hard_acc = hard_accuracy(model, test_dataset)
    return iters, train_loss, train_soft_acc, val_soft_acc,\
          train_hard_acc, val_hard_acc, test_soft_acc, test_hard_acc

def plot_results(results):
    iters, train_loss, train_soft_acc, val_soft_acc,\
          train_hard_acc, val_hard_acc, test_soft_acc, test_hard_acc = results
    
    final_train_soft_acc = train_soft_acc[-1]
    final_val_soft_acc = val_soft_acc[-1]
    final_train_hard_acc = train_hard_acc[-1]
    final_val_hard_acc = val_hard_acc[-1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)
    title = f"Model used: CNN (width={WIDTH}, batch norm={USE_BATCH_NORM})\n" +\
    f"Training curve (batch size={BATCH_SIZE}, learning rate={LEARNING_RATE}, num epochs={NUM_EPOCHS})\n" +\
    f"Final Train Soft Acc: {final_train_soft_acc:.3f}, Final Val Soft Acc: {final_val_soft_acc:.3f},\n" +\
    f" Final Train Hard Acc: {final_train_hard_acc:.3f}, Final Val Hard Acc: {final_val_hard_acc:.3f}, \n" +\
    f"Test Soft Acc: {test_soft_acc:.3f}, Test Hard Acc: {test_hard_acc:.3f}, \n" +\
    f"Final loss {train_loss[-1]:.3f}"
    fig.suptitle(title, fontsize=12)

    axes[0].plot(iters[:len(train_loss)], train_loss)
    axes[0].set_title("Loss over iterations")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")

    axes[1].plot(iters[:len(train_soft_acc)], train_soft_acc)
    axes[1].plot(iters[:len(val_soft_acc)], val_soft_acc)
    axes[1].set_title("Soft Accuracy over iterations")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(["Train", "Validation"])

    axes[2].plot(iters[:len(train_hard_acc)], train_hard_acc)
    axes[2].plot(iters[:len(val_hard_acc)], val_hard_acc)
    axes[2].set_title("Hard Accuracy over iterations")
    axes[2].set_xlabel("Iterations")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend(["Train", "Validation"])
    
    plt.savefig(SAVE_DIR / 'training_curves.png')

def main():
    model = CNN(width=WIDTH, bn=USE_BATCH_NORM)
    dataset = LISADataset(DATASET_PATH)
    if not os.path.exists(SAVE_DIR):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    N = len(dataset)
    train_size = int(0.7 * N)
    val_size   = int(0.2 * N)
    test_size  = N - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])
    results = train_model(model, train_dataset, val_dataset, test_dataset,
                        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, 
                        num_epochs=NUM_EPOCHS, plot_every=PLOT_EVERY, print_every=PRINT_EVERY)
    
    torch.save(model.state_dict(), SAVE_DIR / 'model_weights.pth')
    torch.save(results, SAVE_DIR / 'results.pt')
    plot_results(results)

if __name__ == "__main__":
    main()