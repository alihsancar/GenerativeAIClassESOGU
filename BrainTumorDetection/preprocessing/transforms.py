"""
Görüntü ön işleme ve veri artırma dönüşümleri.
Notebook Cell 6, 13, 21'den taşındı — değiştirilmedi.
"""

import numpy as np
from PIL import Image
import torchvision.transforms as T

from config.settings import MEAN, STD


# ── Auto-Crop ─────────────────────────────────────────────────────────────────
def auto_crop_brain(pil_img: Image.Image) -> Image.Image:
    """
    Eşik tabanlı maskeleme ile beyin dokusunu kırpar, arka planı kaldırır.
    Notebook Cell 6 ile birebir aynıdır.
    """
    gray = np.array(pil_img.convert("L"))
    mask = gray > 10
    if not mask.any():
        return pil_img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad  = 5
    rmin = max(0, rmin - pad);  rmax = min(gray.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad);  cmax = min(gray.shape[1] - 1, cmax + pad)
    return pil_img.crop((cmin, rmin, cmax + 1, rmax + 1))


# ── Eğitim dönüşümleri ────────────────────────────────────────────────────────
def get_train_transform(size: int = 224) -> T.Compose:
    """
    Eğitim sırasında kullanılan veri artırma pipeline'ı.
    Notebook Cell 6 (224) ve Cell 13 (384) ile aynıdır.
    """
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


# ── Doğrulama / Test dönüşümü ─────────────────────────────────────────────────
def get_val_transform(size: int = 224) -> T.Compose:
    """
    Doğrulama ve test için minimal dönüşüm (augmentation yok).
    Notebook Cell 6 ve 13 ile aynıdır.
    """
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


# ── TTA dönüşümleri ───────────────────────────────────────────────────────────
def make_tta_transforms(size: int) -> list:
    """
    8 adet TTA dönüşümü döner.
    Notebook Cell 21 ile birebir aynıdır.
    """
    return [
        # 1 — Orijinal
        T.Compose([T.Resize((size, size)),
                   T.ToTensor(), T.Normalize(MEAN, STD)]),
        # 2 — Yatay çevirme
        T.Compose([T.Resize((size, size)),
                   T.RandomHorizontalFlip(p=1.0),
                   T.ToTensor(), T.Normalize(MEAN, STD)]),
        # 3 — +10° döndürme
        T.Compose([T.Resize((size, size)),
                   T.RandomRotation((10, 10)),
                   T.ToTensor(), T.Normalize(MEAN, STD)]),
        # 4 — -10° döndürme
        T.Compose([T.Resize((size, size)),
                   T.RandomRotation((-10, -10)),
                   T.ToTensor(), T.Normalize(MEAN, STD)]),
        # 5 — %5 büyütüp merkez kırpma
        T.Compose([T.Resize((int(size * 1.05), int(size * 1.05))),
                   T.CenterCrop(size),
                   T.ToTensor(), T.Normalize(MEAN, STD)]),
        # 6 — +5° + yatay çevirme
        T.Compose([T.Resize((size, size)),
                   T.RandomRotation((5, 5)),
                   T.RandomHorizontalFlip(p=1.0),
                   T.ToTensor(), T.Normalize(MEAN, STD)]),
        # 7 — Parlaklık/kontrast varyasyonu
        T.Compose([T.Resize((size, size)),
                   T.ColorJitter(brightness=0.1, contrast=0.1),
                   T.ToTensor(), T.Normalize(MEAN, STD)]),
        # 8 — %10 büyütüp merkez kırpma
        T.Compose([T.Resize((int(size * 1.1), int(size * 1.1))),
                   T.CenterCrop(size),
                   T.ToTensor(), T.Normalize(MEAN, STD)]),
    ]


# ── Web sitesi için tek görüntü inference dönüşümü ───────────────────────────
def preprocess_single(pil_img: Image.Image, size: int = 384) -> "torch.Tensor":
    """
    Web sitesinden gelen tek bir PIL görüntüsünü modele hazırlar:
    auto_crop → val_transform → (1, 3, size, size) tensor
    """
    import torch
    img = auto_crop_brain(pil_img.convert("RGB"))
    tf  = get_val_transform(size)
    return tf(img).unsqueeze(0)   # batch boyutu ekle
