import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Imports and Setup (Robust)
cell1 = """
# Install required packages if missing
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check for PyTorch Lightning
try:
    import pytorch_lightning
except ImportError:
    print("Installing pytorch_lightning...")
    install("pytorch_lightning")

# Check for Seaborn
try:
    import seaborn
except ImportError:
    print("Installing seaborn...")
    install("seaborn")

# Check for PyTDC (tdc) - Optional
try:
    import tdc
except ImportError:
    print("PyTDC not found, but it is optional for this notebook.")

# Check for OmegaConf
try:
    import omegaconf
except ImportError:
    print("Installing omegaconf...")
    install("omegaconf")

# Check for RDKit
try:
    import rdkit
except ImportError:
    print("Installing rdkit...")
    install("rdkit")

# Imports
import torch
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from tqdm import tqdm

# Fix path
sys.path.append(os.getcwd())

# ConPLex imports
from conplex_dti.dataset.datamodules import get_task_dir, DTIDataModule
from conplex_dti.featurizer import get_featurizer
from conplex_dti.model.architectures import SimpleCoembedding, DeepCoembedding, ResidualCoembedding, CrossAttentionCoembedding

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Plotting Style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
"""

# Cell 2: Load Data
cell2 = """
print("Loading BIOSNAP test set...")
task_dir = get_task_dir("biosnap", database_root="datasets")
drug_featurizer = get_featurizer("MorganFeaturizer", save_dir=task_dir)
target_featurizer = get_featurizer("ProtBertFeaturizer", save_dir=task_dir)

datamodule = DTIDataModule(
    task_dir,
    drug_featurizer,
    target_featurizer,
    device=device,
    batch_size=32,
    shuffle=False,
    num_workers=0
)
datamodule.prepare_data()
datamodule.setup()
test_loader = datamodule.test_dataloader()
print("Data loaded successfully.")
"""

# Cell 3: Define Models and Hyperparameters
cell3 = """
# Define model paths and their specific hyperparameters
models_config = {
    "Baseline (Simple)": {
        "path": "best_models/biosnap_baseline_pure/biosnap_baseline_pure_best_model.pt",
        "class": SimpleCoembedding,
        "kwargs": {
            "drug_shape": 2048,
            "target_shape": 1024,
            "latent_dimension": 1024
        }
    },
    "Deep MLP": {
        "path": "best_models/biosnap_deep_contrastive/biosnap_deep_contrastive_best_model.pt",
        "class": DeepCoembedding,
        "kwargs": {
            "drug_shape": 2048,
            "target_shape": 1024,
            "latent_dimension": 1024,
            "num_layers": 3,
            "dropout": 0.1
        }
    },
    "Residual": {
        "path": "best_models/biosnap_residual_improved_v2/biosnap_residual_improved_v2_best_model.pt",
        "class": ResidualCoembedding,
        "kwargs": {
            "drug_shape": 2048,
            "target_shape": 1024,
            "latent_dimension": 1024,
            "num_blocks": 2
        }
    },
    "Cross-Attention": {
        "path": "best_models/biosnap_cross_attention_v1/biosnap_cross_attention_v1_best_model.pt",
        "class": CrossAttentionCoembedding,
        "kwargs": {
            "drug_shape": 2048,
            "target_shape": 1024,
            "latent_dimension": 1024,
            "num_heads": 4,
            "dropout": 0.1
        }
    }
}
"""

# Cell 4: Evaluation Function
cell4 = """
def evaluate_model(name, config):
    print(f"Evaluating {name}...")
    model_path = config["path"]
    
    if not os.path.exists(model_path):
        print(f"Error: File not found at {model_path}")
        return None

    # Initialize Model with specific kwargs
    try:
        model = config["class"](**config["kwargs"])
    except Exception as e:
        print(f"Error initializing model {name}: {e}")
        return None
    
    # Load Weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading weights for {name}: {e}")
        return None

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=name):
            drug, target, label = batch
            drug = drug.to(device)
            target = target.to(device)
            
            # Forward pass
            pred = model(drug, target)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.numpy())

    return {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels)
    }

results = {}
for name, config in models_config.items():
    res = evaluate_model(name, config)
    if res:
        results[name] = res
"""

# Cell 5: ROC and PR Curves
cell5 = """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for name, res in results.items():
    # ROC Curve
    fpr, tpr, _ = roc_curve(res["labels"], res["preds"])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(res["labels"], res["preds"])
    pr_auc = average_precision_score(res["labels"], res["preds"])
    ax2.plot(recall, precision, label=f'{name} (AUPR = {pr_auc:.3f})')

# ROC Settings
ax1.plot([0, 1], [0, 1], 'k--', lw=2)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic (ROC)')
ax1.legend(loc="lower right")

# PR Settings
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend(loc="lower left")

plt.tight_layout()
plt.show()
"""

# Cell 6: Confusion Matrices
cell6 = """
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

for i, (name, res) in enumerate(results.items()):
    if i >= 4: break # Limit to 4 plots
    
    # Binarize predictions
    binary_preds = (res["preds"] >= 0.5).astype(int)
    cm = confusion_matrix(res["labels"], binary_preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f'{name}\\nThreshold = 0.5')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
    axes[i].set_xticklabels(['Negative', 'Positive'])
    axes[i].set_yticklabels(['Negative', 'Positive'])

plt.tight_layout()
plt.show()
"""

nb.cells = [
    nbf.v4.new_code_cell(cell1),
    nbf.v4.new_code_cell(cell2),
    nbf.v4.new_code_cell(cell3),
    nbf.v4.new_code_cell(cell4),
    nbf.v4.new_code_cell(cell5),
    nbf.v4.new_code_cell(cell6)
]

with open('ConPLex/biosnap_architecture_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully: ConPLex/biosnap_architecture_comparison.ipynb")
