from tqdm import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(train_dl, model, loss_fn, optimizer):
    model.train()
    running_loss = 0
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
    epoch_loss = (running_loss) / len(train_dl)
    return epoch_loss


def validate_one_epoch(valid_dl, model, loss_fn, optimizer):
    with torch.no_grad():
        model.eval()
        running_loss = 0
        for itr, batch in enumerate(tqdm(valid_dl)):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
        epoch_loss = (running_loss) / len(valid_dl)
    return epoch_loss
