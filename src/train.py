import torch
import numpy as np
from tqdm import tqdm

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w//2, 0, W)
    bby1 = np.clip(cy - cut_h//2, 0, H)
    bbx2 = np.clip(cx + cut_w//2, 0, W)
    bby2 = np.clip(cy + cut_h//2, 0, H)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (W*H))
    return x, y, y[idx], lam

def train_epoch(model, loader, criterion, optimizer, device,
                scheduler=None,
                use_mixup=False, mixup_alpha=0.4,
                use_cutmix=False, cutmix_alpha=1.0,
                clip_grad=2.0):
    model.train()
    total_loss = total_samples = 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_cutmix:
            x1, y1, y2, lam = cutmix_data(imgs.clone(), labels, alpha=cutmix_alpha)
            out = model(x1)
            loss = lam*criterion(out, y1) + (1-lam)*criterion(out, y2)
        elif use_mixup:
            x1, y1, y2, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
            out = model(x1)
            loss = lam*criterion(out, y1) + (1-lam)*criterion(out, y2)
        else:
            out = model(imgs)
            loss = criterion(out, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = imgs.size(0)
        total_loss    += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            if criterion:
                total_loss += criterion(out, labels).item() * imgs.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
    return (total_loss/total if criterion else None), (correct/total)
