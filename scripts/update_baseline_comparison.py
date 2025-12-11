import json

notebook_path = "ConPLex/biosnap_baseline_comparison.ipynb"

with open(notebook_path, "r") as f:
    notebook = json.load(f)

# Find the cell with the model paths and update it
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        new_source = []
        for line in source:
            if '"Original Baseline":' in line:
                new_line = line.replace("baseline_final", "biosnap_baseline_no_contrast")
                new_source.append(new_line)
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(notebook_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Updated {notebook_path}")
