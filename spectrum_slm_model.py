"""
spectrum_slm_model.py
=====================
Spectrum-SLM: Transformer-based Cognitive Radio Spectrum Sensing

Architecture:
  PSD (192,) → PatchEmbedding (192 bins, patch_size=1) → (192, 128)
  + CLS token → (193, 128)
  → FrequencyAwarePositionalEncoding
  → TransformerEncoder (4 layers, 4 heads, d=128, Pre-LN)
  → CLS token → Multi-task heads:
      PUHead:  128→64→2      (binary: PU present/absent)
      ModHead: 128→64→5      (BPSK/QPSK/8PSK/16QAM/DQPSK)
      SNRHead: 128→64→1      (regression, dB)
      GenHead: 128→256→192   (generative: next PSD)
      MSMHead: (192,128)→192×1 (masked spectrum modelling)

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : May 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Patch Embedding  (192 bins → 192 tokens, patch_size=1)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Each frequency bin becomes one spectral token via a learned linear projection.

    Input  : (B, 192)
    Output : (B, 193, d_model)   [192 bin-tokens + 1 prepended CLS token]
    """

    def __init__(self, n_bins: int = 192, patch_size: int = 1, d_model: int = 128):
        super().__init__()
        assert n_bins % patch_size == 0
        self.n_bins    = n_bins
        self.patch_size = patch_size
        self.n_patches = n_bins // patch_size   # 192

        # Linear projection: patch_size → d_model  (1 → 128)
        self.projection = nn.Linear(patch_size, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 192)  →  (B, 193, d_model)"""
        B = x.size(0)
        # (B, 192) → (B, 192, 1) → project → (B, 192, d_model)
        x = x.view(B, self.n_patches, self.patch_size)
        x = self.projection(x)
        # Prepend CLS: (B, 193, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Frequency-Aware Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyAwarePositionalEncoding(nn.Module):
    """
    Blends learned positional embeddings with fixed sinusoidal encodings via
    a learnable scalar alpha. Informs the model of both patch order and
    physical frequency meaning.

    Input/Output: (B, 193, d_model)
    """

    def __init__(self, n_tokens: int = 193, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Learnable positional embedding
        self.pos_emb = nn.Embedding(n_tokens, d_model)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Fixed sinusoidal encoding
        pe       = torch.zeros(n_tokens, d_model)
        position = torch.arange(n_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, 193, d_model)

        # Blend weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len   = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        learned   = self.pos_emb(positions).unsqueeze(0)
        sinus     = self.pe[:, :seq_len, :]
        alpha     = torch.sigmoid(self.alpha)
        return self.dropout(x + alpha * learned + (1 - alpha) * sinus)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transformer Encoder  (Pre-LN for training stability)
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumTransformerEncoder(nn.Module):
    """Input/Output: (B, 193, d_model)"""

    def __init__(self, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,   # Pre-LN
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Task Output Heads
# ─────────────────────────────────────────────────────────────────────────────

class PUDetectionHead(nn.Module):
    """Binary: PU Present (1) / Absent (0)"""
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 2))
    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.net(cls)   # (B, 2)


class ModulationHead(nn.Module):
    """5-class: BPSK=0  QPSK=1  8PSK=2  16QAM=3  DQPSK=4"""
    def __init__(self, d_model=128, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, n_classes))
    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.net(cls)   # (B, n_classes)


class SNRHead(nn.Module):
    """Regression: predict SNR in dB"""
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1))
    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.net(cls).squeeze(-1)   # (B,)


class GenerativeHead(nn.Module):
    """Predict next PSD snapshot (192 bins) from CLS representation."""
    def __init__(self, d_model=128, n_bins=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_bins))
    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.net(cls)   # (B, 192)


class MSMHead(nn.Module):
    """Reconstruct masked PSD bins during Phase 1 pre-training."""
    def __init__(self, d_model=128, patch_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, patch_size))
    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """patch_feats: (B, 192, d_model) → (B, 192, 1)"""
        return self.net(patch_feats)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Model
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumSLM(nn.Module):
    """
    Spectrum-SLM: Small Language Model for Cognitive Radio Spectrum Sensing.

    Token flow:
      PSD (B,192) → PatchEmbed → (B,192,128)
      + CLS       → (B,193,128)
      → PosEnc   → (B,193,128)
      → Encoder  → (B,193,128)
      → CLS[:,0] → (B,128) → [PU | Mod | SNR | Gen] heads
    """

    def __init__(
        self,
        n_bins:          int   = 192,
        patch_size:      int   = 1,
        d_model:         int   = 128,
        nhead:           int   = 4,
        num_layers:      int   = 4,
        dim_feedforward: int   = 512,
        dropout:         float = 0.1,
        n_mod_classes:   int   = 5,
    ):
        super().__init__()
        self.n_bins  = n_bins
        self.d_model = d_model

        self.tokenizer = PatchEmbedding(n_bins, patch_size, d_model)
        self.pos_enc   = FrequencyAwarePositionalEncoding(
            n_tokens = self.tokenizer.n_patches + 1,   # 193
            d_model  = d_model, dropout = dropout)
        self.encoder   = SpectrumTransformerEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout)

        self.pu_head  = PUDetectionHead(d_model)
        self.mod_head = ModulationHead(d_model, n_mod_classes)
        self.snr_head = SNRHead(d_model)
        self.gen_head = GenerativeHead(d_model, n_bins)
        self.msm_head = MSMHead(d_model, patch_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, psd: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_msm: bool = False) -> dict:
        """
        psd        : (B, 192)
        mask       : (B, 192) bool — True = masked token (Phase 1)
        return_msm : also return MSM reconstruction

        Returns dict with keys:
          pu_logits  (B, 2)
          mod_logits (B, 5)
          snr_pred   (B,)
          gen_pred   (B, 192)
          cls_feat   (B, 128)
          msm_pred   (B, 192, 1)  — only if return_msm=True
        """
        tokens = self.tokenizer(psd)      # (B, 193, d)
        tokens = self.pos_enc(tokens)     # (B, 193, d)

        # Apply Phase 1 mask: zero out masked bin-tokens
        if mask is not None:
            B = mask.size(0)
            cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)   # (B, 193)
            tokens[full_mask] = 0.0

        feats    = self.encoder(tokens)   # (B, 193, d)
        cls_feat = feats[:, 0, :]        # (B, d)

        out = {
            'pu_logits' : self.pu_head(cls_feat),
            'mod_logits': self.mod_head(cls_feat),
            'snr_pred'  : self.snr_head(cls_feat),
            'gen_pred'  : self.gen_head(cls_feat),
            'cls_feat'  : cls_feat,
        }
        if return_msm:
            patch_feats    = feats[:, 1:, :]        # (B, 192, d)
            out['msm_pred'] = self.msm_head(patch_feats)   # (B, 192, 1)

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for PU detection with class imbalance."""
    def __init__(self, gamma=2.0, alpha: Optional[torch.Tensor] = None,
                 reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss:
      L = α·Focal(PU) + β·CE(Mod) + γ·Huber(SNR)

    Supports Kendall uncertainty weighting (learn_weights=True).
    """

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3,
                 pu_class_weight: Optional[torch.Tensor] = None,
                 focal_gamma=2.0, learn_weights=False):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.learn_weights = learn_weights

        self.pu_loss_fn  = FocalLoss(focal_gamma, pu_class_weight)
        self.mod_loss_fn = nn.CrossEntropyLoss()
        self.snr_loss_fn = nn.HuberLoss(delta=1.0)    # robust to outliers

        if learn_weights:
            self.log_var_pu  = nn.Parameter(torch.zeros(1))
            self.log_var_mod = nn.Parameter(torch.zeros(1))
            self.log_var_snr = nn.Parameter(torch.zeros(1))

    def forward(self, pu_logits, pu_labels, mod_logits, mod_labels,
                snr_pred, snr_labels) -> Tuple[torch.Tensor, dict]:
        l_pu  = self.pu_loss_fn(pu_logits,  pu_labels)
        l_mod = self.mod_loss_fn(mod_logits, mod_labels)
        l_snr = self.snr_loss_fn(snr_pred,   snr_labels.float())

        if self.learn_weights:
            p_pu  = torch.exp(-self.log_var_pu[0])
            p_mod = torch.exp(-self.log_var_mod[0])
            p_snr = torch.exp(-self.log_var_snr[0])
            total = (p_pu  * l_pu  + self.log_var_pu[0]  +
                     p_mod * l_mod + self.log_var_mod[0] +
                     p_snr * l_snr + self.log_var_snr[0])
        else:
            total = self.alpha * l_pu + self.beta * l_mod + self.gamma * l_snr

        return total, {'total': total.item(), 'pu': l_pu.item(),
                       'mod': l_mod.item(), 'snr': l_snr.item()}


class MSMLoss(nn.Module):
    """MSE reconstruction loss for Masked Spectrum Modelling (Phase 1)."""
    def forward(self, pred: torch.Tensor, true: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        pred, true : (B, 192, 1)
        mask       : (B, 192) bool — True = masked
        """
        diff    = (pred - true) ** 2
        mask_f  = mask.float().unsqueeze(-1)
        return (diff * mask_f).sum() / (mask_f.sum() + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = SpectrumSLM()
    print(f"Parameters: {model.count_parameters():,}")

    B   = 4
    psd = torch.randn(B, 192)
    out = model(psd, return_msm=True)
    for k, v in out.items():
        print(f"  {k:12s}: {v.shape}")

    assert out['pu_logits'].shape  == (B, 2),   "PU head shape mismatch"
    assert out['mod_logits'].shape == (B, 5),   "Mod head shape mismatch"
    assert out['snr_pred'].shape   == (B,),     "SNR head shape mismatch"
    assert out['msm_pred'].shape   == (B, 192, 1), "MSM head shape mismatch"
    print("All shapes correct ✓")
