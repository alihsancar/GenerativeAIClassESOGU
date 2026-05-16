"""
Attention Rollout görselleştirme fonksiyonları.
Notebook Cell 30, 32'den taşındı — değiştirilmedi.
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy import ndimage

from config.settings import DEVICE, ROLLOUT_DISCARD_RATIO


# ── Yardımcı: beyin maskesi ───────────────────────────────────────────────────
def get_brain_mask(img_np: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Normalize edilmiş [0,1] RGB array'den beyin dokusuna ait ikili maske üretir.
    Notebook Cell 30 ile birebir aynıdır.
    """
    gray = img_np.mean(axis=2)
    mask = gray > threshold
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=2)
    mask = ndimage.binary_dilation(mask, iterations=2)
    return mask.astype(float)


# ── Yardımcı: tensor → görüntü ───────────────────────────────────────────────
def inv_normalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Normalize edilmiş tensoru [0,1] aralığında HWC numpy array'e çevirir.
    Notebook Cell 30 ile birebir aynıdır.
    """
    inv = T.Compose([
        T.Normalize([0., 0., 0.], [2., 2., 2.]),
        T.Normalize([-0.5, -0.5, -0.5], [1., 1., 1.]),
    ])
    return inv(tensor).permute(1, 2, 0).numpy().clip(0, 1)


# ── Ana fonksiyon: Attention Rollout ─────────────────────────────────────────
def get_attention_rollout(
    model,
    img_tensor: torch.Tensor,
    device=DEVICE,
    discard_ratio: float = ROLLOUT_DISCARD_RATIO,
) -> tuple:
    """
    ViTForImageClassification (output_attentions=True) modelinden
    Attention Rollout haritası üretir.

    Parametreler
    ------------
    model        : output_attentions=True ile yüklenmiş ViT modeli
    img_tensor   : (C, H, W) tensor — batch boyutu olmadan
    device       : torch.device
    discard_ratio: alt yüzdelik dilimi sıfırla (0.95 → en yüksek %5 kalsın)

    Dönen değer
    -----------
    (amap_resized, pred_idx)
      amap_resized : (H, W) float array, [0,1] normalize, beyin maskeli
      pred_idx     : modelin tahmin ettiği sınıf indeksi
    Notebook Cell 32 ile birebir aynıdır.
    """
    model.eval()
    with torch.no_grad():
        out = model(
            pixel_values=img_tensor.unsqueeze(0).to(device),
            output_attentions=True,
        )

    attentions = out.attentions
    # Her katman: (1, num_heads, seq_len, seq_len) → başlar üzerinden ortalama
    attn_mat = torch.stack([a.mean(dim=1).squeeze(0) for a in attentions])

    # Rollout: katmanlar boyunca özyinelemeli matris çarpımı
    rollout = torch.eye(attn_mat.size(-1), device=attn_mat.device)
    for attn in attn_mat:
        flat   = attn.view(-1)
        thresh = flat.kthvalue(int(flat.size(0) * discard_ratio)).values
        attn_c = attn * (attn > thresh).float()
        attn_c = attn_c + torch.eye(attn_c.size(-1), device=attn_c.device)
        attn_c = attn_c / attn_c.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attn_c, rollout)

    # CLS → patch attention değerleri
    mask = rollout[0, 1:]
    grid = int(mask.size(0) ** 0.5)
    amap = mask.reshape(grid, grid).cpu().float().numpy()
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

    # Görüntü boyutuna yeniden ölçekle ve beyin maskesi uygula
    img_np       = inv_normalize(img_tensor)
    brain_mask   = get_brain_mask(img_np)
    h, w         = img_np.shape[:2]
    amap_resized = np.array(
        Image.fromarray((amap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    amap_resized = amap_resized * brain_mask
    if amap_resized.max() > 0:
        amap_resized = (amap_resized - amap_resized.min()) / (amap_resized.max() - amap_resized.min() + 1e-8)

    pred_idx = out.logits.argmax(1).item()
    return amap_resized, pred_idx


# ── Overlay görüntüsü üretici ─────────────────────────────────────────────────
def make_overlay(img_tensor: torch.Tensor, amap: np.ndarray,
                 alpha: float = 0.45) -> np.ndarray:
    """
    Orijinal MR görüntüsü üzerine ısı haritasını bindirir.

    Parametreler
    ------------
    img_tensor : (C, H, W) normalize tensor
    amap       : (H, W) [0,1] attention haritası
    alpha      : ısı haritası opaklık ağırlığı (0.45 → notebook Cell 33 değeri)

    Dönen değer
    -----------
    np.ndarray (H, W, 3) RGB overlay, [0,1] aralığında
    """
    import matplotlib.pyplot as plt
    img_np = inv_normalize(img_tensor)
    heat   = plt.cm.jet(amap)[:, :, :3]
    return ((1 - alpha) * img_np + alpha * heat).clip(0, 1)


# ── Web sitesi için hepsi bir arada ──────────────────────────────────────────
def rollout_for_web(
    model,
    pil_img: Image.Image,
    val_transform,
    discard_ratio: float = ROLLOUT_DISCARD_RATIO,
) -> dict:
    """
    Web sitesi için tek PIL görüntüsünden rollout üretir.

    Dönen değer
    -----------
    {
        "amap"    : np.ndarray (H, W)  — dikkat haritası
        "overlay" : np.ndarray (H, W, 3) — overlay görüntüsü
        "pred_idx": int
    }
    """
    from preprocessing.transforms import auto_crop_brain
    img      = auto_crop_brain(pil_img.convert("RGB"))
    img_t    = val_transform(img)                      # (C, H, W)
    amap, pred = get_attention_rollout(model, img_t, discard_ratio=discard_ratio)
    overlay    = make_overlay(img_t, amap)
    return {"amap": amap, "overlay": overlay, "pred_idx": pred}
