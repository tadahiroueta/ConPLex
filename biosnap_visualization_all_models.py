import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# Define paths to history files
base_path = "best_models"
model_paths = {
    "Baseline": os.path.join(base_path, "biosnap_baseline_contrastive", "history.json"),
    "Residual": os.path.join(base_path, "biosnap_residual_contrastive", "history.json"),
    "Deep MLP": os.path.join(base_path, "biosnap_deep_contrastive", "history.json"),
    "Cross Attention": os.path.join(base_path, "biosnap_crossattn_contrastive_fixed", "history.json")
}

# Load data
data = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        with open(path, 'r') as f:
            data[name] = json.load(f)
    else:
        print(f"Warning: File not found for {name} at {path}")

# Helper function to extract metrics
def extract_metrics(data_dict, metric_name):
    df_list = []
    for model_name, history in data_dict.items():
        if metric_name in history:
            values = history[metric_name]
            epochs = range(1, len(values) + 1)
            df_list.append(pd.DataFrame({
                "Epoch": epochs,
                "Model": model_name,
                metric_name: values
            }))
    return pd.concat(df_list) if df_list else pd.DataFrame()

# 1. Training Loss Comparison
print("Generating Training Loss Plot...")
df_loss = extract_metrics(data, "train_loss")
if not df_loss.empty:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_loss, x="Epoch", y="train_loss", hue="Model", marker="o")
    plt.title("Training Loss over Epochs (BIOSNAP)")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(title="Model")
    plt.savefig("biosnap_train_loss.png")
    plt.close()

# 2. Validation AUPR Comparison
print("Generating Validation AUPR Plot...")
df_aupr = extract_metrics(data, "val_aupr")
if not df_aupr.empty:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_aupr, x="Epoch", y="val_aupr", hue="Model", marker="o")
    plt.title("Validation AUPR over Epochs (BIOSNAP)")
    plt.ylabel("AUPR")
    plt.xlabel("Epoch")
    plt.legend(title="Model")
    plt.savefig("biosnap_val_aupr.png")
    plt.close()

# 3. Validation AUROC Comparison
print("Generating Validation AUROC Plot...")
df_auroc = extract_metrics(data, "val_auroc")
if not df_auroc.empty:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_auroc, x="Epoch", y="val_auroc", hue="Model", marker="o")
    plt.title("Validation AUROC over Epochs (BIOSNAP)")
    plt.ylabel("AUROC")
    plt.xlabel("Epoch")
    plt.legend(title="Model")
    plt.savefig("biosnap_val_auroc.png")
    plt.close()

# 4. Final Test Results Comparison
print("Generating Test Results Plot...")
test_results = []
for name, history in data.items():
    if "test_results" in history:
        res = history["test_results"]
        test_results.append({
            "Model": name,
            "Test AUPR": res.get("test/aupr", 0),
            "Test AUROC": res.get("test/auroc", 0),
            "Epochs Trained": res.get("epoch", 0)
        })

df_test = pd.DataFrame(test_results)
print("\nFinal Test Results:")
print(df_test)

if not df_test.empty:
    # Bar plot for Test AUPR
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_test, x="Model", y="Test AUPR", palette="viridis")
    plt.title("Final Test AUPR Comparison (BIOSNAP)")
    plt.ylim(0, 1.0)
    plt.savefig("biosnap_test_aupr.png")
    plt.close()

print("\nAnalysis Complete. Plots saved as PNG files.")
