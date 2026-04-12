import sys
from pathlib import Path
import optuna

from model import CNN, LISADataset
from trainer import set_seed, compute_loss, split_dataset

import torch
from torch.utils.data import DataLoader

# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.config import *

DEVICE = torch.device('mps')
TRAIN, VAL, TEST = split_dataset(LISADataset(training_dataset_path))
N_TRIALS = 10

def pseudo_train(model, train_data, val_data,
                learning_rate, batch_size,
                num_epochs):
    model.train()
    model = model.to(DEVICE)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for e in range(num_epochs):
        for batch in train_loader:
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            z = model(images)
            loss = criterion(z, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    valid_loss = compute_loss(model, val_loader, criterion)

    return valid_loss

def objective(trial):
    set_seed(42)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    width = trial.suggest_categorical("width", [4, 6, 8])
    
    model = CNN(width=width, bn=True, normalise=False, drop=0.0)
    
    valid_loss = pseudo_train(
        model, TRAIN, VAL,
        learning_rate=lr, batch_size=64, num_epochs=3)
    
    return valid_loss

def main():
    study = optuna.create_study(direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(objective, n_trials=N_TRIALS)

if __name__ == "__main__":
    main()