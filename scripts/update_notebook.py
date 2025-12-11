import json
import os

notebook_path = 'ConPLex/final_project_comparison.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Update the description cell (first cell)
description_cell = nb['cells'][0]
source = description_cell['source']
# Find where to insert the new model description
insert_idx = -1
for i, line in enumerate(source):
    if "Training: 50 Epochs, Early Stopping (enabled), ReduceLROnPlateau" in line:
        insert_idx = i + 1
        break

if insert_idx != -1:
    new_lines = [
        "\n",
        "3.  **Cross-Attention (`biosnap_cross_attention_v1`):**\n",
        "    *   Architecture: CrossAttentionCoembedding (Drug attends to Target, 4 heads)\n",
        "    *   Training: 50 Epochs, Early Stopping, ReduceLROnPlateau"
    ]
    # Check if already added to avoid duplication
    if "Cross-Attention" not in "".join(source):
        source[insert_idx:insert_idx] = new_lines

# Update the model_paths cell (second cell, execution_count 1)
code_cell = nb['cells'][1]
source = code_cell['source']
# Find the model_paths definition
for i, line in enumerate(source):
    if '"Improved Residual": os.path.join(base_path, "biosnap_residual_improved_v2", "history.json")' in line:
        # Check if next line already has Cross-Attention
        if i + 1 < len(source) and "Cross-Attention" in source[i+1]:
            break
        
        # Add comma to the current line if needed (though python dict doesn't strictly need it if it's not the last, but for valid syntax in list)
        # Actually the line probably doesn't have a comma at the end if it was the last item.
        # Let's just replace the line with comma and add new line.
        source[i] = '    "Improved Residual": os.path.join(base_path, "biosnap_residual_improved_v2", "history.json"),\n'
        source.insert(i + 1, '    "Cross-Attention": os.path.join(base_path, "biosnap_cross_attention_v1", "history.json")\n')
        break

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
