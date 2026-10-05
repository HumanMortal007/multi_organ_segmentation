"""
datasets/synapse_dataset.py
============================
PyTorch Dataset for Synapse/BTCV multi-organ segmentation.

Training data  : 2D slices in data/Synapse/train_npz/ (.npz files)
                 Keys: "image" (H,W) float32 [0,1], "label" (H,W) float32.
Test data      : 3D volumes in data/Synapse/test_vol_h5/ (.npy.h5 files)
                 Keys: "image" (D,H,W), "label" (D,H,W).

9 Classes: 0=background, 1=aorta, 2=gallbladder, 3=spleen,
           4=left_kidney, 5=right_kidney, 6=liver, 7=stomach, 8=pancreas
"""

import os
import random
import numpy as np
import torch
import h5py
from scipy.ndimage import zoom, rotate as ndrotate
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Augmentation helpers  (matching TransUNet implementation)
# ============================================================

def random_rot_flip(image, label):
    """Random 90-degree rotation (0-3x) + random horizontal/vertical flip."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """Small rotation in [-20, 20] degrees. Uses order=0 for label."""
    angle = np.random.randint(-20, 20)
    image = ndrotate(image, angle, order=3, reshape=False)
    label = ndrotate(label, angle, order=0, reshape=False)
    return image, label


# ============================================================
# Transform
# ============================================================

class RandomGenerator:
    """
    Matches TransUNet RandomGenerator exactly.
    Resizes to output_size, applies optional augmentation.
    Returns: image [1,H,W] FloatTensor, label [H,W] LongTensor
    """
    def __init__(self, output_size=(224, 224), augment=True,
                 do_rot_flip=True, do_rotate=True):
        self.output_size = output_size
        self.augment = augment
        self.do_rot_flip = do_rot_flip
        self.do_rotate = do_rotate

    def __call__(self, sample):
        image = sample["image"]  # (H, W) float32
        label = sample["label"]  # (H, W) float32 integer class ids

        if self.augment:
            if self.do_rot_flip and random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif self.do_rotate and random.random() > 0.5:
                image, label = random_rotate(image, label)

        x, y = image.shape[:2]
        oh, ow = self.output_size
        if x != oh or y != ow:
            image = zoom(image, (oh / x, ow / y), order=3)
            label = zoom(label, (oh / x, ow / y), order=0)

        image_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # [1,H,W]
        label_t = torch.from_numpy(label.astype(np.float32)).long()        # [H,W]
        return {"image": image_t, "label": label_t}


# ============================================================
# Training Dataset  (2D slices from .npz)
# ============================================================

class SynapseTrainDataset(Dataset):
    """2D CT slice dataset for training/validation."""

    def __init__(self, data_dir, list_dir, transform=None):
        list_file = os.path.join(list_dir, "train.txt")
        with open(list_file, "r") as f:
            self.sample_list = [ln.strip() for ln in f if ln.strip()]
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sn = self.sample_list[idx]
        path = os.path.join(self.data_dir, sn + ".npz")
        data = np.load(path)
        image = data["image"].astype(np.float32)  # (H, W)
        label = data["label"].astype(np.float32)  # (H, W) class ids

        sample = {"image": image, "label": label}
        if self.transform is not None:
            result = self.transform(sample)
        else:
            result = {
                "image": torch.from_numpy(image).unsqueeze(0),
                "label": torch.from_numpy(label).long(),
            }
        result["case_name"] = sn
        return result


# ============================================================
# Test Dataset  (3D volumes from .npy.h5)
# ============================================================

class SynapseTestDataset(Dataset):
    """3D CT volume dataset for evaluation."""

    def __init__(self, data_dir, list_dir):
        list_file = os.path.join(list_dir, "test_vol.txt")
        with open(list_file, "r") as f:
            self.sample_list = [ln.strip() for ln in f if ln.strip()]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        vn = self.sample_list[idx]
        path = os.path.join(self.data_dir, f"{vn}.npy.h5")
        with h5py.File(path, "r") as hf:
            image = hf["image"][:]  # (D, H, W)
            label = hf["label"][:]  # (D, H, W)
        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "case_name": vn,
        }


# ============================================================
# Factory functions
# ============================================================

def get_train_loader(cfg, augment=True):
    """Build training DataLoader."""
    img_size = cfg["dataset"]["img_size"]
    transform = RandomGenerator(
        output_size=(img_size, img_size),
        augment=augment and cfg["augmentation"]["enabled"],
        do_rot_flip=cfg["augmentation"]["random_rot_flip"],
        do_rotate=cfg["augmentation"]["random_rotate"],
    )
    dataset = SynapseTrainDataset(
        data_dir=cfg["dataset"]["train_data_dir"],
        list_dir=cfg["dataset"]["list_dir"],
        transform=transform,
    )
    nw = cfg["training"]["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=cfg["training"]["pin_memory"],
        persistent_workers=(nw > 0 and cfg["training"]["persistent_workers"]),
        drop_last=True,
    )
    return loader, dataset


def get_test_loader(cfg):
    """Build test DataLoader (batch_size=1 for volumetric evaluation)."""
    dataset = SynapseTestDataset(
        data_dir=cfg["dataset"]["test_data_dir"],
        list_dir=cfg["dataset"]["list_dir"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    return loader, dataset
