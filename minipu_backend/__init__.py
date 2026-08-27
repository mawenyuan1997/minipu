import torch
import os

import minipu_backend._C

torch.rename_privateuse1_backend("magic_gpu")

print("[minipu] Backend initialized successfully!")
