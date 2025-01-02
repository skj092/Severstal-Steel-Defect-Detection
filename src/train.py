from dataset import SteelDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from model import Unet
from torch.optim import Adam
from engine import train_one_epoch, validate_one_epoch
import torch.nn as nn
import torch


if __name__ == "__main__":
    # Loading Data
    path = Path("/home/sonujha/rnd/Severstal-Steel-Defect-Detection/data/")
    df = pd.read_csv(path / "train.csv")
    df = df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
    df["defects"] = df.count(axis=1)
    df = df.sample(n=10)

    # Cross Validation
    train_df, valid_df = train_test_split(
        df, test_size=0.2, stratify=df["defects"], random_state=42
    )

    # Dataset and DataLoader
    train_ds = SteelDataset(train_df, path / "train_images")
    valid_ds = SteelDataset(valid_df, path / "train_images")

    train_dl = DataLoader(
        train_ds, batch_size=4, num_workers=2, pin_memory=True, shuffle=True
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=4, num_workers=2, pin_memory=True, shuffle=False
    )

    # Model, loss function and optimizer
    model = Unet("resnet18", encoder_weights="imagenet",
                 classes=4, activation=None)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    for epoch in range(2):
        state = {
            'epoch': epoch,
            'best_loss': best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        train_loss = train_one_epoch(
            train_dl, model, loss_fn=loss_fn, optimizer=optimizer)
        valid_loss = validate_one_epoch(
            valid_dl, model, loss_fn=loss_fn, optimizer=optimizer)
        print(f"Epoch {epoch} train_loss {train_loss}, valid_loss {valid_loss}")

        # save the model
        if valid_loss < best_loss:
            print('=========New optimal fouund, saving state =========')
            state['best_loss'] = best_loss = valid_loss
            torch.save(state, 'model.pth')
