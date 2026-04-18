"""QuantaBurst — Training script (384-channel from scratch)."""
import os, sys, random, math, time
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pytorch_msssim import ms_ssim

from net import QuantaBurst
from data import PhotonEventDataset
from loss import SmoothL1Loss, PerceptualLoss, GradientLoss, FFTL1Loss

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT      = "/mnt/zone/A/dataset"
OUT_DIR        = "/mnt/zone/A/QuantaBurst/output"
SEED           = 42
NUM_EPOCHS     = 200
BATCH_SIZE     = 1
CROP_SIZE      = 384
NUM_WORKERS    = 4
GRAD_CLIP_NORM = 1.0
AUG_PROB       = 0.2

# Loss weights
W_CHARB   = 0.50
W_MSSSIM  = 0.30
W_VGG     = 0.10
W_EDGE    = 0.10
W_FFT     = 0.05

LR        = 2e-4
ETA_MIN   = 1e-6

# Loss phasing (Charbonnier-only until PSNR gate, then ramp perceptual losses)
PSNR_GATE     = 34.0  # Charb-only until best PSNR exceeds this (epoch-independent)
RAMP_EPOCHS   = 15    # epochs to linearly ramp perceptual losses to full weight

# EMA
EMA_DECAY = 0.999

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── WandB ────────────────────────────────────────────────────────────────────
try:
    wandb.init(
        project="spc-quantaburst",
        name="spc-quantaburst-384ch",
        settings=wandb.Settings(init_timeout=300),
        config=dict(
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, crop_size=CROP_SIZE,
            lr=LR, cosine_t_max=NUM_EPOCHS, eta_min=ETA_MIN,
            psnr_gate=PSNR_GATE, ramp_epochs=RAMP_EPOCHS,
            w_charb=W_CHARB, w_msssim=W_MSSSIM, w_vgg=W_VGG, w_edge=W_EDGE, w_fft=W_FFT,
            ema_decay=EMA_DECAY, grad_clip=GRAD_CLIP_NORM,
        ),
    )
    _WANDB_OK = True
except Exception as e:
    print(f"[wandb] init failed ({e}), continuing without wandb")
    _WANDB_OK = False

# ── Seed ──────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
train_dataset = PhotonEventDataset(DATA_ROOT, split='train', crop_size=CROP_SIZE, augment=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuantaBurst().to(device)

param_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[model] QuantaBurst — {param_count:.2f}M parameters on {device}")

# ── EMA shadow ────────────────────────────────────────────────────────────────
ema_shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

def update_ema(model, shadow, decay):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            shadow[k].mul_(decay).add_(v.float(), alpha=1.0 - decay)

# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN,
)

scaler = GradScaler()

# ── Losses ────────────────────────────────────────────────────────────────────
charb_loss = SmoothL1Loss().to(device)
vgg_loss   = PerceptualLoss().to(device)
edge_loss  = GradientLoss().to(device)
fft_loss   = FFTL1Loss().to(device)

# ── Checkpoint helpers ────────────────────────────────────────────────────────
os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

def save_ckpt(path, epoch, best_psnr):
    torch.save({
        "epoch": epoch,
        "best_psnr": best_psnr,
        "model": model.state_dict(),
        "ema": {k: v.half() for k, v in ema_shadow.items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)

# ── Resume — load latest checkpoint (epoch_* files take priority over best_model) ──────────────
import glob as _glob
start_epoch = 0
best_psnr   = 0.0

_ckpt_dir = os.path.join(OUT_DIR, "checkpoints")
_epoch_ckpts = sorted(_glob.glob(os.path.join(_ckpt_dir, "epoch_*.pth")))
_best_ckpt   = os.path.join(_ckpt_dir, "best_model.pth")
resume_path  = _epoch_ckpts[-1] if _epoch_ckpts else (_best_ckpt if os.path.isfile(_best_ckpt) else None)

def _filter_trunk(sd):
    """Strip any non-trunk.* keys (e.g. novel-module keys from QuantaBurst_novel).
    Returns a dict with keys matching QuantaBurst's own state-dict."""
    own_keys = set(model.state_dict().keys())
    # Keep keys that exist in our model; remap 'trunk.*' in case source used same prefix
    out = {}
    for k, v in sd.items():
        if k in own_keys:
            out[k] = v
    return out

if resume_path:
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    raw_sd  = ckpt.get("model", ckpt)
    raw_ema = ckpt.get("ema", {})

    filtered_sd  = _filter_trunk(raw_sd)
    filtered_ema = _filter_trunk(raw_ema)

    own_keys = set(model.state_dict().keys())
    missing  = own_keys - set(filtered_sd.keys())
    extra    = set(raw_sd.keys()) - own_keys

    missing_msg = f", {len(missing)} missing (random-init)" if missing else ""
    extra_msg   = f", {len(extra)} skipped (novel-module keys)" if extra else ""
    print(f"[resume] {os.path.basename(resume_path)}  "
          f"loaded {len(filtered_sd)}/{len(own_keys)} keys{missing_msg}{extra_msg}")

    model.load_state_dict(filtered_sd, strict=False)

    # Only restore optimizer/scheduler if checkpoint matches our model exactly
    # (i.e. came from a previous base-QuantaBurst run, not from novel)
    if not extra and not missing:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr   = ckpt["best_psnr"]
        print(f"[resume] full state restored — epoch {start_epoch}, best {best_psnr:.2f} dB")
    else:
        # Cross-model load: start fresh optimizer/scheduler but keep the PSNR bar
        # Rebuild scheduler at epoch-80 position (~1.31e-4 LR) without triggering warnings
        best_psnr   = ckpt["best_psnr"]
        start_epoch = 80
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN, last_epoch=80
        )
        print(f"[resume] cross-model checkpoint — optimizer reset, "
              f"scheduler at epoch 80 (lr={optimizer.param_groups[0]['lr']:.2e}), "
              f"best_psnr={best_psnr:.2f} dB")

    for k in ema_shadow:
        if k in filtered_ema:
            ema_shadow[k].copy_(filtered_ema[k].float())
    ema_loaded = len([k for k in ema_shadow if k in filtered_ema])
    print(f"[resume] EMA: {ema_loaded}/{len(ema_shadow)} keys loaded")
else:
    print("[train] no checkpoint found, starting from epoch 0")

# ── PSNR helper ───────────────────────────────────────────────────────────────
def calc_psnr(pred, target):
    mse = (pred - target).pow(2).mean()
    if mse < 1e-10:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)

# ── Training loop ─────────────────────────────────────────────────────────────
steps_per_epoch = len(train_loader)
print(f"[train] epochs {start_epoch}→{NUM_EPOCHS}, "
      f"{len(train_dataset)} samples, {steps_per_epoch} steps/epoch")
print(f"[train] Charbonnier-only, lr={LR}, cosine T_max={NUM_EPOCHS}")
print("-" * 80)

ramp_start_epoch = -1   # set when best_psnr first crosses PSNR_GATE

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    t_charb = t_ssim = t_vgg = t_edge = t_fft = t_total = 0.0
    t_psnr = 0.0
    t0 = time.time()

    for step, (inp, gt) in enumerate(train_loader):
        inp = inp.to(device, non_blocking=True)
        gt  = gt.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(inp)

            lc = charb_loss(pred, gt)
            ls = lv = le = lf = torch.zeros(1, device=device)

            loss = lc

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        update_ema(model, ema_shadow, EMA_DECAY)

        with torch.no_grad():
            t_charb += lc.item()
            t_ssim  += ls.item()
            t_vgg   += lv.item()
            t_edge  += le.item()
            t_fft   += lf.item()
            t_total += loss.item()
            t_psnr  += calc_psnr(pred.detach().float().clamp(0, 1), gt).item()

        if (step + 1) % 50 == 0:
            done = step + 1
            print(f"    ep {epoch+1} step {done:4d}/{steps_per_epoch}  "
                  f"loss {t_total/done:.4f}  psnr {t_psnr/done:.2f} dB  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}",
                  flush=True)

    # Epoch stats
    n = steps_per_epoch
    avg_psnr  = t_psnr / n
    avg_loss  = t_total / n
    avg_ssim  = 1.0 - (t_ssim / n)
    elapsed   = time.time() - t0
    current_lr = optimizer.param_groups[0]["lr"]

    is_best = avg_psnr > best_psnr
    if is_best:
        best_psnr = avg_psnr

    scheduler.step()

    star = " ★" if is_best else ""
    print(f"  {epoch+1:3d}/{NUM_EPOCHS}  "
          f"psnr {avg_psnr:.2f} dB  ssim {avg_ssim:.4f}  "
          f"loss {avg_loss:.4f}  "
          f"[c {t_charb/n:.4f}]  "
          f"lr {current_lr:.2e}  {elapsed:.0f}s{star}")

    if _WANDB_OK:
        wandb.log({
            "train/psnr":        avg_psnr,
            "train/ssim":        avg_ssim,
            "train/loss":        avg_loss,
            "train/loss_charb":  t_charb / n,
            "train/loss_ssim":   t_ssim  / n,
            "train/loss_vgg":    t_vgg   / n,
            "train/loss_edge":   t_edge  / n,
            "train/loss_fft":    t_fft   / n,
            "train/lr":          current_lr,
            "train/best_psnr":   best_psnr,
            "epoch":             epoch + 1,
        }, step=epoch + 1)

    # Save best
    if is_best:
        save_ckpt(os.path.join(OUT_DIR, "checkpoints", "best_model.pth"),
                  epoch, best_psnr)

    # Periodic save every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_ckpt(os.path.join(OUT_DIR, "checkpoints", f"epoch_{epoch+1:03d}.pth"),
                  epoch, best_psnr)

print(f"\n[done] best psnr = {best_psnr:.2f} dB")
