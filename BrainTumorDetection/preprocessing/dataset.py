"""
Özel Dataset sınıfları.
Notebook Cell 6'dan taşındı — değiştirilmedi.
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder

from config.settings import SEED, BATCH_224, BATCH_384
from preprocessing.transforms import (
    auto_crop_brain,
    get_train_transform,
    get_val_transform,
)


# ── AutoCropImageFolder ───────────────────────────────────────────────────────
class AutoCropImageFolder(Dataset):
    """
    ImageFolder üzerine auto_crop_brain uygular.
    Notebook Cell 6 ile birebir aynıdır.
    """
    def __init__(self, root: str, transform=None):
        self.dataset      = ImageFolder(root)
        self.transform    = transform
        self.classes      = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples      = self.dataset.samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        img = auto_crop_brain(img)
        if self.transform:
            img = self.transform(img)
        return img, label


# ── SubsetDataset ─────────────────────────────────────────────────────────────
class SubsetDataset(Dataset):
    """
    Bir Dataset'ten verilen indeks listesine göre alt küme oluşturur.
    Notebook Cell 6 ile birebir aynıdır.
    """
    def __init__(self, ds: Dataset, indices: list):
        self.ds      = ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.ds[self.indices[i]]


# ── DataLoader fabrikası ──────────────────────────────────────────────────────
def build_dataloaders(train_dir: str, test_dir: str, size: int = 224):
    """
    Eğitim, doğrulama ve test DataLoader'larını döner.
    Notebook Cell 6 ve 13 mantığı — değiştirilmedi.

    Parametreler
    ------------
    train_dir : str   Kaggle 'Training/' klasörü
    test_dir  : str   Kaggle 'Testing/' klasörü
    size      : int   224 veya 384

    Dönen değer
    -----------
    train_loader, val_loader, test_loader, val_indices
    """
    batch = BATCH_224 if size == 224 else BATCH_384

    train_tf = get_train_transform(size)
    val_tf   = get_val_transform(size)

    full_train = AutoCropImageFolder(train_dir, transform=train_tf)
    val_size   = int(0.15 * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_ds_clean = AutoCropImageFolder(train_dir, transform=val_tf)
    val_indices  = val_ds.indices
    val_dataset  = SubsetDataset(val_ds_clean, val_indices)
    test_dataset = AutoCropImageFolder(test_dir, transform=val_tf)

    kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds,    batch_size=batch, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_dataset, batch_size=batch, shuffle=False, **kwargs)
    test_loader  = DataLoader(test_dataset,batch_size=batch, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, val_indices
