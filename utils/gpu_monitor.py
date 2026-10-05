"""
utils/gpu_monitor.py
=====================
GPU memory monitoring utilities.
"""

import torch


def get_gpu_info():
    """Return dict with GPU name, total VRAM, allocated, max allocated (in GB)."""
    if not torch.cuda.is_available():
        return {"available": False}

    idx = torch.cuda.current_device()
    total   = torch.cuda.get_device_properties(idx).total_memory / 1e9
    alloc   = torch.cuda.memory_allocated(idx)   / 1e9
    max_alloc = torch.cuda.max_memory_allocated(idx) / 1e9
    name    = torch.cuda.get_device_name(idx)
    return {
        "available"  : True,
        "name"       : name,
        "total_gb"   : total,
        "alloc_gb"   : alloc,
        "max_alloc_gb": max_alloc,
    }


def print_gpu_info(prefix=""):
    info = get_gpu_info()
    if not info["available"]:
        print(f"{prefix}CUDA not available.")
        return
    print(f"{prefix}GPU   : {info['name']}")
    print(f"{prefix}VRAM  : {info['alloc_gb']:.2f} GB / {info['total_gb']:.2f} GB")
    print(f"{prefix}Max   : {info['max_alloc_gb']:.2f} GB")


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
