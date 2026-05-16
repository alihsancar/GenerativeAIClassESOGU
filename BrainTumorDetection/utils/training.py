"""
Generic eğitim döngüsü.
Notebook Cell 14'ten taşındı — değiştirilmedi.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import ViTForImageClassification, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from config.settings import DEVICE, WEIGHT_DECAY, WARMUP_RATIO


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    patience: int,
    criterion,
    save_path: str,
    tag: str = "Model",
) -> tuple:
    """
    Herhangi bir ViT veya ViT_GRU modelini eğitir.

    Parametreler
    ------------
    model        : ViTForImageClassification veya ViT_GRU
    train_loader : eğitim DataLoader
    val_loader   : doğrulama DataLoader
    epochs       : maksimum epoch sayısı
    lr           : öğrenme oranı
    patience     : early stopping sabrı
    criterion    : FocalLoss veya başka kayıp fonksiyonu
    save_path    : en iyi model .pth kayıt yolu
    tag          : loglarda görünecek model ismi

    Dönen değer
    -----------
    (best_state_dict, history_dict, best_val_acc)
    Notebook Cell 14 ile birebir aynıdır.
    """
    optimizer    = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler()

    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state   = None
    patience_cnt = 0

    is_vit = isinstance(model, ViTForImageClassification)

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        tl = tc = tt = 0
        for imgs, lbls in tqdm(train_loader, desc=f"{tag} Ep {epoch+1}/{epochs} [T]", leave=False):
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                logits = model(pixel_values=imgs).logits if is_vit else model(imgs)
                loss   = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer);  scaler.update();  scheduler.step()
            tl += loss.item() * imgs.size(0)
            tc += (logits.argmax(1) == lbls).sum().item()
            tt += imgs.size(0)

        # ── Validation ──
        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"{tag} Ep {epoch+1}/{epochs} [V]", leave=False):
                imgs = imgs.to(DEVICE, non_blocking=True)
                lbls = lbls.to(DEVICE, non_blocking=True)
                with autocast(dtype=torch.bfloat16):
                    logits = model(pixel_values=imgs).logits if is_vit else model(imgs)
                    loss   = criterion(logits, lbls)
                vl += loss.item() * imgs.size(0)
                vc += (logits.argmax(1) == lbls).sum().item()
                vt += imgs.size(0)

        t_acc = tc / tt * 100
        v_acc = vc / vt * 100
        history["train_loss"].append(tl / tt)
        history["val_loss"].append(vl / vt)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            mark = " ⭐ BEST"
        else:
            patience_cnt += 1
            mark = f" (patience {patience_cnt}/{patience})"

        print(
            f"{tag} Epoch {epoch+1:2d}/{epochs}  "
            f"Train: {tl/tt:.4f}/{t_acc:.2f}%  "
            f"Val: {vl/vt:.4f}/{v_acc:.2f}%{mark}"
        )

        if patience_cnt >= patience:
            print(f"Early stopping — {epoch+1} epoch.")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, save_path)
    print(f"✅ {tag} Best Val Acc: {best_val_acc:.2f}% → {save_path}")
    return best_state, history, best_val_acc
