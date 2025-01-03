import torch
import numpy as np
import logging
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)


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


def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculate dice of positive and negative images seperately'''
    '''probability and truth must be torch tensor'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert (probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)

        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


def predict(X, threshold):
    '''X is sigmoid outout of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] == 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred = c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=[1]):
    '''computes mean iou for batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs)
    labels = np.array(labels)
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


class Meter(object):
    '''A meter to keep track of iou and dice scores through an epoch'''

    def __init__(self):
        self.base_threshold = 0.5
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(
            probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metric(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou


def epoch_log(epoch_loss, meter):
    '''loggint the metrics at the end of an epoch'''
    dices, iou = meter.get_metric()
    dice, dice_neg, dice_pos = dices
    logging.info(f"Loss: {epoch_loss} | IoU {iou} | dice {dice} | dice_neg {dice_neg} | dice_pos {dice_pos}")
    return dice, iou
