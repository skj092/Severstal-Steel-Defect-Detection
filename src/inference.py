import albumentations as A
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import Unet
from albumentations.pytorch import ToTensorV2


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def get_transforms(train, mean, std):
    list_transforms = []
    list_transforms.extend(
        [
            A.Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ]
    )
    list_tfms = A.Compose(list_transforms)
    return list_tfms


class SteelDataset(Dataset):
    def __init__(self, df, data_folder):
        self.df = df
        self.root_dir = data_folder
        self.transform = get_transforms(True, mean, std)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id, rle, class_id = df.iloc[idx]
        image_path = os.path.join(self.root_dir, img_id)
        img = np.array(Image.open(image_path))
        # (256, 1600) and (1, 256, 1600, 4)
        augmented = self.transform(image=img)
        img = augmented["image"]
        return img


if __name__ == "__main__":
    path = Path('/home/sonujha/rnd/Severstal-Steel-Defect-Detection/data/')
    df = pd.read_csv(path/'sample_submission.csv')
    ds = SteelDataset(df, path/'test_images')
    dl = DataLoader(ds, batch_size=16, shuffle=False, pin_memory=True)

    # Load the model
    model = Unet("resnet18", encoder_weights="imagenet",
                 classes=4, activation=None)
    state = torch.load('model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state['state_dict'])
    print("==== model loaded =====")

    # prediction on single image
    img = ds[0]
    img = img[None, :, :, :]  # batch of size 1
    output = model(img)
    preds = torch.sigmoid(output)
