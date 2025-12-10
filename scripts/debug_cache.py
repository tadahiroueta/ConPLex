import h5py
import pandas as pd
from pathlib import Path

def sanitize_string(s):
    return s.replace("/", "|")

data_dir = Path("datasets/DUDe")
df = pd.read_csv(data_dir / "full.tsv", sep="\t")
smiles = df["Molecule_SMILES"].iloc[0]
print(f"First SMILES: {smiles}")
sanitized = sanitize_string(smiles)
print(f"Sanitized: {sanitized}")

h5_path = data_dir / "Morgan_features.h5"
with h5py.File(h5_path, "r") as f:
    print(f"Keys in H5: {len(f.keys())}")
    if sanitized in f:
        print("Found in H5!")
    else:
        print("NOT found in H5!")
        # Print first 5 keys
        print("First 5 keys in H5:")
        for i, k in enumerate(f.keys()):
            if i >= 5: break
            print(k)
