import torch


def initialize_device(preferred: str = None) -> torch.device:
    """`initialize_device` function sets up device for tensors - and uses"""
    if preferred:
        return torch.device(preferred)

    print(f"CUDA version: {torch.version.cuda}")

    is_cuda = torch.cuda.is_available()
    print(f"Is CUDA supported by this system? {is_cuda}")
    if is_cuda:
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    device = torch.device("cuda") if is_cuda else torch.device("cpu")
    print(f"Chose: {device}")
    return device
