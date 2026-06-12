import torch


def get_device() -> torch.device:
    """
    Returns the best available device.
    On Apple Silicon, this prefers MPS.
    Falls back to CUDA, then CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[device] Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[device] Using CPU")
    return device
