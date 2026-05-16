"""
Proje genelinde kullanılan sabitler ve yapılandırma.
Notebook'taki SEED, DEVICE, CLASSES, MEAN/STD vb. değerleri buraya taşındı.
"""

import torch

# ── Tekrarlanabilirlik ─────────────────────────────────────────────────────────
SEED = 42

# ── Cihaz ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Sınıflar (Kaggle Brain Tumor MRI Dataset — sorted) ────────────────────────
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS2IDX   = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS     = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ── Görüntü normalizasyon ──────────────────────────────────────────────────────
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

# ── Model isimleri (HuggingFace) ──────────────────────────────────────────────
MODEL_NAMES = {
    "l224": "google/vit-large-patch16-224",
    "b384": "google/vit-base-patch16-384",
    "l384": "google/vit-large-patch16-384",
    "lgru": "google/vit-large-patch16-384",   # GRU backbone da L/384
}

# ── Model dosya yolları (.pth) ─────────────────────────────────────────────────
# Web sitesinde kullanırken burası güncellenir
MODEL_PATHS = {
    "l224": "weights/model_l224_best.pth",
    "b384": "weights/model_b384_best.pth",
    "l384": "weights/model_l384_best.pth",
    "lgru": "weights/model_lgru_best.pth",
}

# ── Eğitim parametreleri ───────────────────────────────────────────────────────
BATCH_224    = 64
BATCH_384    = 16
EPOCHS       = 25
LR           = 2e-5
LR_LARGE     = 1e-5      # ViT-L/16 @ 384 için
LR_BACKBONE  = 1e-5      # GRU backbone
LR_HEAD      = 5e-5      # GRU head
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE     = 6

# ── Focal Loss sınıf ağırlıkları ──────────────────────────────────────────────
# Sıra: CLASSES ile aynı → glioma=2.5, meningioma=1.5, notumor=1.0, pituitary=1.0
FOCAL_ALPHA         = [2.5, 1.5, 1.0, 1.0]
FOCAL_GAMMA         = 2.0
LABEL_SMOOTHING     = 0.1

# ── GRU parametreleri ─────────────────────────────────────────────────────────
GRU_HIDDEN  = 512
GRU_LAYERS  = 2
GRU_DROPOUT = 0.3

# ── Attention Rollout ─────────────────────────────────────────────────────────
ROLLOUT_DISCARD_RATIO = 0.95

# ── TTA dönüşüm sayısı ────────────────────────────────────────────────────────
TTA_N_TRANSFORMS = 8

# ── Web sitesi inference — hangi model kullanılacak ──────────────────────────
#   "ensemble"  → 4 modelin softmax ortalaması  (en kapsamlı, en yavaş)
#   "l384"      → ViT-L/16 @ 384 tek model      (en iyi tek model, hızlı)
#   "b384"      → ViT-B/16 @ 384                (küçük GPU'lar için)
INFERENCE_MODE = "l384"   # Web sitesi için önerilen seçim (aşağıya bak)
