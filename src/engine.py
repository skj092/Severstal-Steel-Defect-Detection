from tqdm import tqdm
import torch
from utils import compute_iou_batch, Meter, epoch_log

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(train_dl, model, loss_fn, optimizer):
    model.train()
    running_loss = 0
    iou = 0
    for itr, batch in enumerate(tqdm(train_dl)):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        outputs = outputs.detach().cpu()
        iou += compute_iou_batch(outputs, targets)
    epoch_loss = (running_loss) / len(train_dl)
    epoch_iou = iou / len(train_dl)

    return epoch_loss, epoch_iou


def validate_one_epoch(valid_dl, model, loss_fn, optimizer):
    model.eval()
    meter = Meter()
    running_loss = 0
    iou = 0
    for itr, batch in enumerate(tqdm(valid_dl)):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = loss_fn(outputs, targets)
        running_loss += loss.item()
        outputs = outputs.detach().cpu()
        iou += compute_iou_batch(outputs, targets)
        meter.update(targets, outputs)
    epoch_loss = (running_loss) / len(valid_dl)
    epoch_iou = iou / len(valid_dl)
    epoch_log(epoch_loss, meter)
    return epoch_loss, epoch_iou
