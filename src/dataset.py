from torch.utils.data import Dataset, DataLoader
from utils import img_mask_pair
import os
from pathlib import Path
import pandas as pd
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
import numpy as np

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def get_transforms(train, mean, std):
    list_transforms = []
    if train:
        list_transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
            ]
        )
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
        img_id, mask = img_mask_pair(idx, self.df)
        image_path = os.path.join(self.root_dir, img_id)
        img = np.array(Image.open(image_path))
        # (256, 1600) and (1, 256, 1600, 4)
        augmented = self.transform(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        mask = mask.permute(2, 0, 1)  # (4, 256, 1600)
        return img, mask


if __name__ == "__main__":
    path = Path("/home/sonujha/rnd/Severstal-Steel-Defect-Detection/data/")
    df = pd.read_csv(path / "train.csv")
    df = df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
    ds = SteelDataset(df, path)
    x, y = ds[0]
    print(x.shape, y.shape)
