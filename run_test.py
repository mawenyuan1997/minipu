import torch
import minipu_backend 

device = torch.device("minipu:0")
print(f"Custom Accelerator Available: {torch.accelerator.is_available()}")

x = torch.empty(1024, device=device)
y = torch.empty(1024, device=device)

print(f"Tensor X device: {x.device}")

z = x + y

torch.accelerator.synchronize(device)

print("Done")
