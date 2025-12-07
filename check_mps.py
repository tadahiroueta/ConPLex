import torch
import platform

print(f"PyTorch Version: {torch.__version__}")
print(f"Platform: {platform.platform()}")

if torch.backends.mps.is_available():
    print("MPS is available!")
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"Successfully created tensor on device: {x.device}")
else:
    print("MPS is NOT available.")
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("Using CPU.")
