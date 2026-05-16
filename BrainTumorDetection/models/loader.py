"""
Kaydedilmiş .pth ağırlıklarını yükleyen yardımcı fonksiyonlar.
Notebook Cell 15, 17, 24, 25, 26, 31'den taşındı.
"""

import torch
from transformers import ViTForImageClassification

from config.settings import NUM_CLASSES, MODEL_NAMES, MODEL_PATHS, DEVICE
from models.vit_gru import ViT_GRU


def load_vit(model_key: str, output_attentions: bool = False) -> ViTForImageClassification:
    """
    HuggingFace ViTForImageClassification yükler ve .pth ağırlıklarını uygular.

    Parametreler
    ------------
    model_key         : "l224" | "b384" | "l384"
    output_attentions : True ise Attention Rollout için dikkat matrislerini döner

    Notebook Cell 15, 17, 24, 25, 31 mantığı.
    """
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAMES[model_key],
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        output_attentions=output_attentions,
    )
    state = torch.load(MODEL_PATHS[model_key], map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_vit_gru() -> ViT_GRU:
    """
    ViT_GRU modelini yükler.
    Notebook Cell 26 mantığı.
    """
    model = ViT_GRU(num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATHS["lgru"], map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_model_for_web(mode: str = "l384"):
    """
    Web sitesi için tek noktadan model yükleme.

    mode
    ----
    "l384"    → ViT-L/16 @ 384  (önerilen — en iyi tek model, %96.00)
    "b384"    → ViT-B/16 @ 384  (hafif GPU için)
    "l224"    → ViT-L/16 @ 224
    "lgru"    → ViT-L/16 + GRU
    "ensemble"→ tüm 4 modeli döner (dict)

    Dönen değer
    -----------
    Tek model: (model, model_key)
    Ensemble : {"l224": m1, "b384": m2, "l384": m3, "lgru": m4}
    """
    if mode == "ensemble":
        return {
            "l224": load_vit("l224"),
            "b384": load_vit("b384"),
            "l384": load_vit("l384"),
            "lgru": load_vit_gru(),
        }
    elif mode in ("l224", "b384", "l384"):
        return load_vit(mode), mode
    elif mode == "lgru":
        return load_vit_gru(), mode
    else:
        raise ValueError(f"Bilinmeyen mod: {mode}. 'l384', 'b384', 'l224', 'lgru', 'ensemble' kullanın.")
