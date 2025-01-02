import torch
import numpy as np


# RLE-Mask utility function
def img_mask_pair(idx, df):
    """ """
    img_id = df.iloc[idx].name
    labels = df.iloc[idx][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32)
    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            position = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros((256 * 1600), dtype=np.uint8)
            for pos, le in zip(position, length):
                mask[pos: (pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order="F")
    return img_id, masks


def mask2rle(mask):
    """
    mask: np.array (256, 1600, 4)
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


