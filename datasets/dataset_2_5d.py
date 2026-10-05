import os
import random
import numpy as np
import torch
import h5py
from scipy.ndimage import zoom, rotate as ndrotate
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Augmentation helpers for 2.5D (image is 3, H, W)
# ============================================================

def random_rot_flip_2_5d(image, label):
    """Random 90-degree rotation + flip for [3, H, W] image and [H, W] label."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k, axes=(0, 1))
    
    axis = np.random.randint(0, 2)
    # image axes: 0=C, 1=H, 2=W -> flip on axis+1
    image = np.flip(image, axis=axis+1).copy()
    # label axes: 0=H, 1=W -> flip on axis
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate_2_5d(image, label):
    """Small rotation in [-20, 20] degrees for [3, H, W] image."""
    angle = np.random.randint(-20, 20)
    image = ndrotate(image, angle, axes=(1, 2), order=3, reshape=False)
    label = ndrotate(label, angle, axes=(0, 1), order=0, reshape=False)
    return image, label

class RandomGenerator25D:
    def __init__(self, output_size=(224, 224), augment=True,
                 do_rot_flip=True, do_rotate=True):
        self.output_size = output_size
        self.augment = augment
        self.do_rot_flip = do_rot_flip
        self.do_rotate = do_rotate

    def __call__(self, sample):
        image = sample["image"]  # (3, H, W) float32
        label = sample["label"]  # (H, W) float32 integer class ids

        if self.augment:
            if self.do_rot_flip and random.random() > 0.5:
                image, label = random_rot_flip_2_5d(image, label)
            elif self.do_rotate and random.random() > 0.5:
                image, label = random_rotate_2_5d(image, label)

        _, x, y = image.shape
        oh, ow = self.output_size
        if x != oh or y != ow:
            image = zoom(image, (1.0, oh / x, ow / y), order=3)
            label = zoom(label, (oh / x, ow / y), order=0)

        image_t = torch.from_numpy(image.astype(np.float32))       # [3,H,W]
        label_t = torch.from_numpy(label.astype(np.float32)).long()# [H,W]
        return {"image": image_t, "label": label_t}

# ============================================================
# Training Dataset (2.5D: z-1, z, z+1)
# ============================================================

class SynapseTrainDataset25D(Dataset):
    def __init__(self, data_dir, list_dir, transform=None):
        list_file = os.path.join(list_dir, "train.txt")
        with open(list_file, "r") as f:
            self.sample_list = [ln.strip() for ln in f if ln.strip()]
        self.data_dir = data_dir
        self.transform = transform
        
        # Precompute the existence of slices for fast boundary handling
        self.available_slices = set(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sn = self.sample_list[idx]
        case_name, slice_str = sn.split('_slice')
        z = int(slice_str)
        
        prev_sn = f"{case_name}_slice{z-1:03d}"
        next_sn = f"{case_name}_slice{z+1:03d}"
        
        # Boundary replication strategy: if z-1 or z+1 doesn't exist, use z
        if prev_sn not in self.available_slices:
            prev_sn = sn
        if next_sn not in self.available_slices:
            next_sn = sn
            
        path_prev = os.path.join(self.data_dir, prev_sn + ".npz")
        path_z = os.path.join(self.data_dir, sn + ".npz")
        path_next = os.path.join(self.data_dir, next_sn + ".npz")
        
        data_prev = np.load(path_prev)
        data_z = np.load(path_z)
        data_next = np.load(path_next)
        
        # Stack images [3, H, W]
        img_prev = data_prev["image"].astype(np.float32)
        img_z = data_z["image"].astype(np.float32)
        img_next = data_next["image"].astype(np.float32)
        
        image_stack = np.stack([img_prev, img_z, img_next], axis=0)
        
        # Target is just the middle slice mask
        label = data_z["label"].astype(np.float32)
        
        sample = {"image": image_stack, "label": label}
        
        if self.transform is not None:
            result = self.transform(sample)
        else:
            result = {
                "image": torch.from_numpy(image_stack),
                "label": torch.from_numpy(label).long(),
            }
        result["case_name"] = sn
        return result

# ============================================================
# Factory functions
# ============================================================

def get_train_loader_2_5d(cfg, augment=True):
    img_size = cfg["dataset"]["img_size"]
    transform = RandomGenerator25D(
        output_size=(img_size, img_size),
        augment=augment and cfg["augmentation"]["enabled"],
        do_rot_flip=cfg["augmentation"]["random_rot_flip"],
        do_rotate=cfg["augmentation"]["random_rotate"],
    )
    dataset = SynapseTrainDataset25D(
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
