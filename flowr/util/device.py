"""Device utilities for automatic CUDA/MPS/CPU detection and handling."""

import os
from typing import Optional

import torch

_DEVICE = None


def _device_override() -> Optional[torch.device]:
    """The ``FLOWR_DEVICE`` escape hatch, or None when it is unset."""
    requested = os.environ.get("FLOWR_DEVICE")
    return torch.device(requested) if requested else None


def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Priority order:
    1. ``FLOWR_DEVICE``, when set (e.g. ``FLOWR_DEVICE=mps``)
    2. CUDA (if available)
    3. CPU (fallback)

    Apple's MPS backend is NOT auto-selected, even though this is the machine class
    where it exists. Two measured reasons:

    * The ``e3nn`` backbone builds its spherical harmonics in float64
      (``flowr/models/sph.py``), a dtype MPS cannot represent at all, so it raises.
    * Where MPS does run, it does not agree with CPU. Replaying one captured
      ``LigandCFM`` forward pass on an M4: CPU vs CPU is bit-identical (0.0), while
      CPU vs MPS differs by up to 1.2e-3 relative on the bond logits -- and
      generation feeds that output back in for 20-100 sequential integration steps.

    A silently wrong answer is worse than a slower right one, so CPU is the default
    on macOS and MPS is opt-in for those who want to experiment with it.

    Returns:
        torch.device: The selected device.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    override = _device_override()
    if override is not None:
        _DEVICE = override
    elif torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    else:
        _DEVICE = torch.device("cpu")

    return _DEVICE


def get_device_string() -> str:
    """
    Get the device string (e.g., 'cuda', 'mps', 'cpu').

    Returns:
        str: The device string.
    """
    return str(get_device())


def resolve_device(args=None) -> torch.device:
    """Pick the compute device for an inference entrypoint.

    Precedence, highest first:

    1. ``FLOWR_DEVICE`` -- an explicit override (e.g. ``FLOWR_DEVICE=mps``).
    2. CUDA, when the machine has it and ``args.gpus`` is non-zero. ``--gpus`` is a
       device *count*, so ``--gpus 0`` asks for CPU even on a CUDA box.
    3. CPU.

    As in :func:`get_device`, Apple's MPS backend is not auto-selected; see that
    docstring for the measurements behind that choice. On macOS the supported path
    is CPU, and MPS is opt-in via ``FLOWR_DEVICE=mps``.

    Every entrypoint defaults ``--gpus`` to a non-zero count, so on a CUDA machine
    an unmodified command line resolves to ``cuda`` -- exactly what the previous
    hard-coded ``.to("cuda")`` did.
    """
    override = _device_override()
    if override is not None:
        return override
    gpus = getattr(args, "gpus", 1) if args is not None else 1
    if gpus and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_map_location() -> torch.device:
    """
    Get the appropriate map_location for torch.load.

    This should be used when loading checkpoints to ensure they
    are loaded onto the correct device.

    Returns:
        torch.device: The device for map_location.
    """
    return get_device()


def to_device(data, device=None):
    """
    Move data to the specified device (or best available if not specified).

    Args:
        data: Can be a tensor, dict of tensors, or list of tensors.
        device: Target device. If None, uses get_device().

    Returns:
        Data moved to the device.
    """
    if device is None:
        device = get_device()

    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    else:
        return data


def dict_to_device(data_dict: dict, device=None) -> dict:
    """
    Move all tensors in a dictionary to the specified device.

    Args:
        data_dict: Dictionary potentially containing tensors.
        device: Target device. If None, uses get_device().

    Returns:
        dict: Dictionary with tensors moved to device.
    """
    if device is None:
        device = get_device()

    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data_dict.items()}


def clear_cache():
    """Clear GPU memory cache for the current device (CUDA or MPS)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return torch.backends.mps.is_available()


def print_device_info():
    """Print information about the detected device."""
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(
            f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    elif device.type == "mps":
        print("  Apple Silicon (MPS) backend")
    else:
        print("  CPU fallback")
