"""
Sınıf ağırlıklı Focal Loss + label smoothing.
Notebook Cell 9 ile birebir aynıdır — değiştirilmedi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING, DEVICE


class FocalLoss(nn.Module):
    """
    Focal Loss = -α_t · (1 - p_t)^γ · log(p_t)

    Parametreler
    ------------
    gamma            : odaklanma parametresi (varsayılan 2.0)
    label_smoothing  : etiket yumuşatma katsayısı (varsayılan 0.1)
    alpha            : her sınıf için ağırlık listesi
                       [glioma=2.5, meningioma=1.5, notumor=1.0, pituitary=1.0]
    """

    def __init__(
        self,
        gamma: float          = FOCAL_GAMMA,
        label_smoothing: float = LABEL_SMOOTHING,
        alpha: list           = FOCAL_ALPHA,
        device                = DEVICE,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.alpha           = torch.tensor(alpha, device=device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_cls = logits.size(1)
        smooth  = self.label_smoothing / (num_cls - 1)

        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.label_smoothing) + smooth * (1 - one_hot)

        log_p   = F.log_softmax(logits, dim=1)
        p       = log_p.exp()
        focal_w = (1 - p) ** self.gamma
        alpha_t = self.alpha[targets].unsqueeze(1)

        loss = -(alpha_t * focal_w * one_hot * log_p).sum(dim=1)
        return loss.mean()
