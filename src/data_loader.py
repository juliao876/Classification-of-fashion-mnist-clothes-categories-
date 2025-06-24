from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, TensorDataset
import numpy as np
import pandas as pd
import torch

def get_dataloaders(batch_size: int = 64,
                    data_dir:  str = "data/",
                    val_frac:  float = 0.1):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.1)
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full     = datasets.FashionMNIST(root=data_dir, train=True,
                                     download=True, transform=train_tf)
    val_size = int(len(full) * val_frac)
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size])
    val_ds.dataset.transform = test_tf

    test_ds  = datasets.FashionMNIST(root=data_dir, train=False,
                                     download=True, transform=test_tf)

    # oversampling klas problematycznych
    y_tr     = np.array([y for _,y in train_ds])
    weights  = np.ones_like(y_tr, dtype=float)
    for cls in (0,2,3,4,6):  # T-shirt, Pullover, Dress, Coat, Shirt
        weights[y_tr == cls] *= 2.0
    sampler = WeightedRandomSampler(weights,
                                    num_samples=len(weights),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader

def get_csv_test_loader(csv_path: str, batch_size: int = 64):
    df = pd.read_csv(csv_path)
    X = df.iloc[:,1:].values.astype(np.float32)/255.0
    y = df.iloc[:,0].values.astype(np.int64)
    X = X.reshape(-1,1,28,28)
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=False
    )
