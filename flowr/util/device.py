"""Device utilities for automatic CUDA/MPS/CPU detection and handling."""

import torch

_DEVICE = None


def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Priority order:
    1. CUDA (if available)
    2. MPS (Apple Silicon, if available)
    3. CPU (fallback)

    Returns:
        torch.device: The best available device.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
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
