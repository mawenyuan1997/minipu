import torch

def get_device():
    return torch.device("minipu:0")
