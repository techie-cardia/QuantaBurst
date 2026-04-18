"""Loss functions for QuantaBurst training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SmoothL1Loss(nn.Module):
    """Charbonnier loss — smooth L1 approximation robust to outliers."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class PerceptualLoss(nn.Module):
    """Multi-scale VGG feature matching loss."""
    _WEIGHTS = {"r1": 0.2, "r2": 0.3, "r3": 0.5}

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        feats = list(vgg.features.children())
        self.stage1 = nn.Sequential(*feats[:5]).eval()
        self.stage2 = nn.Sequential(*feats[5:10]).eval()
        self.stage3 = nn.Sequential(*feats[10:17]).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # Target activations need no grad — saves significant activation memory
        with torch.no_grad():
            s1_t = self.stage1(target)
            s2_t = self.stage2(s1_t)
            s3_t = self.stage3(s2_t)
        s1_p = self.stage1(pred)
        s2_p = self.stage2(s1_p)
        s3_p = self.stage3(s2_p)
        return (self._WEIGHTS["r1"] * self.criterion(s1_p, s1_t) +
                self._WEIGHTS["r2"] * self.criterion(s2_p, s2_t) +
                self._WEIGHTS["r3"] * self.criterion(s3_p, s3_t))


class GradientLoss(nn.Module):
    """Edge-aware gradient loss — penalises blurry boundaries."""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        def sobel(x):
            dx = x[:, :, :, :-1] - x[:, :, :, 1:]
            dy = x[:, :, :-1, :] - x[:, :, 1:, :]
            return dx, dy
        pdx, pdy = sobel(pred)
        tdx, tdy = sobel(target)
        return self.criterion(pdx, tdx) + self.criterion(pdy, tdy)


class FFTL1Loss(nn.Module):
    """Frequency-domain L1 loss on normalized FFT magnitudes."""
    def forward(self, pred, target):
        H, W = pred.shape[-2], pred.shape[-1]
        # Normalize by spatial size so magnitudes are comparable to pixel-space losses
        pred_fft   = torch.fft.fft2(pred).abs()   / (H * W)
        target_fft = torch.fft.fft2(target).abs() / (H * W)
        return F.l1_loss(pred_fft, target_fft)
