
import torch
import torch.nn as nn
import numpy as np
import time

print("Starting test...")
device = torch.device("mps")
print(f"Using device: {device}")

class ModernBaseline(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        dropout=0.1,
        classify=True,
    ):
        super().__init__()
        self.do_classify = classify
        
        print("Init drug projector...")
        self.drug_projector = nn.Sequential(
            nn.Linear(drug_shape, latent_dimension),
            nn.LayerNorm(latent_dimension), 
            nn.GELU(),
            nn.Dropout(dropout)
        )

        print("Init target projector...")
        self.target_projector = nn.Sequential(
            nn.Linear(target_shape, latent_dimension),
            nn.LayerNorm(latent_dimension),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        print("Init done.")

    def forward(self, drug, target):
        return torch.tensor(0.0)

print("Instantiating model...")
model = ModernBaseline()
print("Model instantiated.")

print("Moving to device...")
start = time.time()
model = model.to(device)
print(f"Moved to device in {time.time() - start:.4f}s")
print("Test complete.")
