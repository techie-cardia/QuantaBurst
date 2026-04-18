"""Inference — generate test predictions from best checkpoint.

Tiled inference: channel-attention operates independently per tile,
so patches can be stitched without blending artifacts.
"""
import os, torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.amp import autocast

from net import QuantaBurst
from data import PhotonEventDataset

DATA_ROOT  = "/mnt/zone/A/dataset"
CKPT_PATH  = "/mnt/zone/A/QuantaBurst/output/checkpoints/best_model.pth"
OUT_DIR    = "/mnt/zone/A/QuantaBurst/output/predictions"
TILE_SIZE  = 400   # process in 400×400 tiles
TILE_PAD   = 16    # overlap padding to avoid any edge effects

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = QuantaBurst().to(device)
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd = ckpt.get("ema", ckpt.get("model", ckpt))
sd = {k: v.float() for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
model.eval()
print(f"[infer] loaded checkpoint, best_psnr={ckpt.get('best_psnr', '?')}")


def infer_tiled(model, inp, tile=TILE_SIZE, pad=TILE_PAD):
    """
    inp: (1, C, H, W) on device
    Returns: (1, 3, H, W) float32 on CPU
    """
    _, C, H, W = inp.shape
    out = torch.zeros(1, 3, H, W, dtype=torch.float32)

    ys = list(range(0, H, tile))
    xs = list(range(0, W, tile))

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)

            # padded region (clamped to image bounds)
            py0 = max(y0 - pad, 0)
            px0 = max(x0 - pad, 0)
            py1 = min(y1 + pad, H)
            px1 = min(x1 + pad, W)

            patch = inp[:, :, py0:py1, px0:px1]
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_patch = model(patch)
            pred_patch = pred_patch.float().clamp(0, 1)

            # crop back to the unpadded region
            cy0 = y0 - py0
            cx0 = x0 - px0
            cy1 = cy0 + (y1 - y0)
            cx1 = cx0 + (x1 - x0)
            out[:, :, y0:y1, x0:x1] = pred_patch[:, :, cy0:cy1, cx0:cx1].cpu()

    return out


# Dataset
test_dataset = PhotonEventDataset(DATA_ROOT, split='test')
os.makedirs(OUT_DIR, exist_ok=True)

with torch.no_grad():
    for i in range(len(test_dataset)):
        inp, _ = test_dataset[i]
        scene = test_dataset.samples[i]['scene']
        frame = test_dataset.samples[i]['frame']

        inp = inp.unsqueeze(0).to(device)
        pred = infer_tiled(model, inp)

        pred_np = (pred.squeeze(0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        out_scene = os.path.join(OUT_DIR, scene)
        os.makedirs(out_scene, exist_ok=True)
        Image.fromarray(pred_np).save(os.path.join(out_scene, f"{frame}.png"))

        print(f"  [{i+1:>3}/{len(test_dataset)}] {scene}/{frame}.png")

print(f"[done] {len(test_dataset)} predictions → {OUT_DIR}")
