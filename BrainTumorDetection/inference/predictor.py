"""
Inference motoru: TTA predict, ensemble ve web için tek görüntü tahmini.
Notebook Cell 22, 23, 24, 25, 26, 27'den taşındı — değiştirilmedi.
"""

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast
from transformers import ViTForImageClassification

from config.settings import NUM_CLASSES, IDX2CLS, DEVICE
from preprocessing.transforms import auto_crop_brain, make_tta_transforms, preprocess_single
from models.vit_gru import ViT_GRU


# ── TTA inference (test seti üzerinde toplu) ──────────────────────────────────
def tta_predict(model, test_paths: list, size: int, is_gru: bool = False) -> np.ndarray:
    """
    Her test görüntüsü için TTA softmax ortalaması döner.

    Parametreler
    ------------
    model      : yüklenmiş ViTForImageClassification veya ViT_GRU
    test_paths : [(img_path, label), ...] — AutoCropImageFolder.samples formatı
    size       : 224 veya 384
    is_gru     : True ise model(pixel_values) değil, model(imgs) çağrılır

    Dönen değer
    -----------
    np.ndarray  shape: (N, NUM_CLASSES) — softmax olasılıkları
    Notebook Cell 22 ile birebir aynı.
    """
    tta_list = make_tta_transforms(size)
    model.eval()
    probs_all = []

    for path, _ in test_paths:
        pil_img  = auto_crop_brain(Image.open(path).convert("RGB"))
        prob_sum = torch.zeros(NUM_CLASSES)
        for tf in tta_list:
            img_t = tf(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad(), autocast(dtype=torch.bfloat16):
                if is_gru:
                    logits = model(img_t)
                else:
                    logits = model(pixel_values=img_t).logits
            prob_sum += torch.softmax(logits.float().squeeze(0).cpu(), dim=0)
        probs_all.append((prob_sum / len(tta_list)).numpy())

    return np.array(probs_all)


# ── Ensemble: 4 modelin softmax ortalaması ───────────────────────────────────
def ensemble_predict(models_dict: dict, test_paths: list) -> np.ndarray:
    """
    4 modelin TTA softmax çıkışlarını eşit ağırlıklı ortalar.

    Parametreler
    ------------
    models_dict : {"l224": model, "b384": model, "l384": model, "lgru": model}
    test_paths  : [(img_path, label), ...]

    Dönen değer
    -----------
    np.ndarray  shape: (N, NUM_CLASSES)
    Notebook Cell 27 ile birebir aynı.
    """
    size_map = {"l224": 224, "b384": 384, "l384": 384, "lgru": 384}
    gru_map  = {"l224": False, "b384": False, "l384": False, "lgru": True}

    all_probs = []
    for key, model in models_dict.items():
        probs = tta_predict(model, test_paths, size=size_map[key], is_gru=gru_map[key])
        all_probs.append(probs)

    # Eşit ağırlıklı ortalama
    ensemble_probs = np.stack(all_probs, axis=0).mean(axis=0)
    return ensemble_probs


# ── Web sitesi: tek PIL görüntüsü → tahmin ───────────────────────────────────
def predict_single(
    model,
    pil_img: Image.Image,
    size: int = 384,
    is_gru: bool = False,
    use_tta: bool = True,
) -> dict:
    """
    Web sitesinden gelen tek bir görüntü üzerinde tahmin yapar.

    Parametreler
    ------------
    model    : yüklenmiş model
    pil_img  : PIL.Image.Image
    size     : 224 veya 384 (modele göre)
    is_gru   : ViT_GRU modeli ise True
    use_tta  : TTA kullanılsın mı (web için genelde True)

    Dönen değer
    -----------
    {
        "label"       : "glioma" | "meningioma" | "notumor" | "pituitary",
        "label_idx"   : int,
        "confidence"  : float (0–1),
        "probs"       : {sınıf: olasılık, ...},
    }
    """
    model.eval()
    img = auto_crop_brain(pil_img.convert("RGB"))

    if use_tta:
        tta_list = make_tta_transforms(size)
        prob_sum = torch.zeros(NUM_CLASSES)
        for tf in tta_list:
            img_t = tf(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad(), autocast(dtype=torch.bfloat16):
                if is_gru:
                    logits = model(img_t)
                else:
                    logits = model(pixel_values=img_t).logits
            prob_sum += torch.softmax(logits.float().squeeze(0).cpu(), dim=0)
        probs = (prob_sum / len(tta_list)).numpy()
    else:
        img_t = preprocess_single(img, size).to(DEVICE)
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            if is_gru:
                logits = model(img_t)
            else:
                logits = model(pixel_values=img_t).logits
        probs = torch.softmax(logits.float().squeeze(0).cpu(), dim=0).numpy()

    pred_idx = int(probs.argmax())
    return {
        "label"      : IDX2CLS[pred_idx],
        "label_idx"  : pred_idx,
        "confidence" : float(probs[pred_idx]),
        "probs"      : {IDX2CLS[i]: float(probs[i]) for i in range(NUM_CLASSES)},
    }
