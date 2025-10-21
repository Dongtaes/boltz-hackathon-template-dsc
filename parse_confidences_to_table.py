import os
import json
import pandas as pd
from pathlib import Path

# === CONFIG ===
ROOT_DIR = Path("hackathon_data/intermediate_files/asos_public/predictions")  # e.g. "./boltz_results/"
OUTPUT_CSV = "merged_confidence_table.csv"

def parse_json_file(json_path: Path):
    """Load a single JSON and flatten nested dicts."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten nested dicts (chains_ptm, pair_chains_iptm)
    flat_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                if isinstance(subv, dict):
                    for subsubk, subsubv in subv.items():
                        flat_data[f"{k}_{subk}_{subsubk}"] = subsubv
                else:
                    flat_data[f"{k}_{subk}"] = subv
        else:
            flat_data[k] = v
    return flat_data

def parse_folder_name(folder_name: str):
    """
    Example: 'boltz_results_1AXB_ORTHOSTERIC_FOS_config_0'
    Extract protein_id, binder_type, ligand_id.
    """
    # Split and find key parts
    parts = folder_name.split("_")
    # example pattern: boltz_results_1AXB_ORTHOSTERIC_FOS_config_0
    protein_id = parts[2]
    binder_type = parts[3].lower()  # orthosteric / allosteric
    ligand_id = parts[4] if len(parts) > 4 else "unknown"
    return protein_id, binder_type, ligand_id

def collect_jsons(root_dir: Path):
    records = []

    for folder in root_dir.glob("boltz_results_*"):
        if not folder.is_dir():
            continue

        protein_id, binder_type, ligand_id = parse_folder_name(folder.name)

        pred_dir = folder / "predictions" / f"{protein_id}_{binder_type.upper()}_{ligand_id}_config_0"
        if not pred_dir.exists():
            print(f"⚠️ Skipping missing folder: {pred_dir}")
            continue

        for json_file in pred_dir.glob("confidence_*.json"):
            model_name = json_file.stem  # e.g. confidence_1AXB_ORTHOSTERIC_FOS_config_0_model_0
            try:
                model_id = int(model_name.split("_model_")[-1])
            except:
                model_id = None

            try:
                flat = parse_json_file(json_file)
                flat.update({
                    "protein_id": protein_id,
                    "binder_type": binder_type,
                    "ligand_id": ligand_id,
                    "model_id": model_id,
                    "json_path": str(json_file),
                })
                records.append(flat)
            except Exception as e:
                print(f"❌ Failed to parse {json_file}: {e}")

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = collect_jsons(ROOT_DIR)
    print(f"✅ Parsed {len(df)} JSON files from {ROOT_DIR}")

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Saved table -> {OUTPUT_CSV}")

    # Show preview
    print(df.head())
