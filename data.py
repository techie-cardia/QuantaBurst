"""Photon binary event dataset — on-the-fly raw npy loading, last 128 frames."""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import random


class PhotonEventDataset(Dataset):
    """
    Streams raw single-photon binary frames on-the-fly.
    Input:  (384, H, W) float32 — 128 temporal frames × 3 color channels.
    Target: (3, H, W) float32 — ground truth RGB in [0, 1].
    """
    NUM_FRAMES = 128

    def __init__(self, data_root, split='train', crop_size=512, augment=True):
        self.data_root = Path(data_root)
        self.split = split
        self.crop_size = crop_size if split == 'train' else None
        self.augment = augment and split == 'train'

        if split == 'test':
            base = self.data_root / 'test'
        else:
            base = self.data_root / 'train'

        scenes = sorted([d.name for d in base.iterdir() if d.is_dir()])

        self.samples = []
        for scene in scenes:
            scene_dir = base / scene
            npy_files = sorted(scene_dir.glob('[0-9]*.npy'))
            for npy_path in npy_files:
                if '_sum' in npy_path.stem:
                    continue
                gt_path = npy_path.with_suffix('.png')
                if split == 'test' or gt_path.exists():
                    self.samples.append({
                        'npy': npy_path,
                        'gt': gt_path if split != 'test' else None,
                        'scene': scene,
                        'frame': npy_path.stem,
                    })

        print(f"PhotonEventDataset [{split}]: {len(self.samples)} samples from {len(scenes)} scenes")

    def __len__(self):
        return len(self.samples)

    def _unpack_frames(self, npy_path):
        """Load last NUM_FRAMES frames, unpack bits, reshape to (384, H, W)."""
        raw = np.load(str(npy_path), mmap_mode='r')[-self.NUM_FRAMES:]
        raw = np.array(raw)
        unpacked = np.unpackbits(raw, axis=2)
        H, W = unpacked.shape[1], unpacked.shape[2]
        x = unpacked.transpose(1, 2, 0, 3).reshape(H, W, -1).transpose(2, 0, 1)
        return x.astype(np.float32)

    def _load_target(self, gt_path):
        img = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inp = self._unpack_frames(sample['npy'])

        if sample['gt'] is not None:
            gt = self._load_target(sample['gt'])
        else:
            gt = np.zeros((3, inp.shape[1], inp.shape[2]), dtype=np.float32)

        if self.crop_size and self.crop_size < inp.shape[1]:
            _, H, W = inp.shape
            y = random.randint(0, H - self.crop_size)
            x = random.randint(0, W - self.crop_size)
            inp = inp[:, y:y + self.crop_size, x:x + self.crop_size]
            gt  = gt[:,  y:y + self.crop_size, x:x + self.crop_size]

        inp = np.ascontiguousarray(inp)
        gt  = np.ascontiguousarray(gt)

        return torch.from_numpy(inp), torch.from_numpy(gt)
