import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

def rank_asos_predictions_with_model(prediction_dir: Path, model_path="ranking_model.pkl", return_dataframe=False):
    """
    Rank Boltz predictions (model_0–model_4) using a trained ML ranking model.

    Args:
        prediction_dir (str or Path): Path to folder containing JSONs like
            confidence_<...>_model_0.json, ..., confidence_<...>_model_4.json
        model_path (str): Path to the trained ranking model (created by train_ranking()).
        return_dataframe (bool): If True, return full DataFrame with scores and ranks.

    Returns:
        list[int]: Model IDs sorted from best (rank 0) to worst (rank 4)
    """
    json_files = sorted(prediction_dir.glob("confidence_*_model_*.json"))
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found in {prediction_dir}")

    # Load trained model and feature columns
    model, feature_cols = joblib.load(model_path)

    # --- Parse JSON files ---
    records = []
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        model_id = int(jf.stem.split("_model_")[-1])

        # Flatten JSON keys (supporting nested dicts like 'chains_ptm' or 'pair_chains_iptm')
        flat = {}
        for k, v in data.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    if isinstance(subv, dict):
                        for subsubk, subsubv in subv.items():
                            flat[f"{k}_{subk}_{subsubk}"] = subsubv
                    else:
                        flat[f"{k}_{subk}"] = subv
            else:
                flat[k] = v

        flat["model_id"] = model_id
        records.append(flat)

    df = pd.DataFrame(records)

    # --- Align columns with trained model ---
    missing_cols = [c for c in feature_cols if c not in df.columns]
    for c in missing_cols:
        df[c] = 0.0  # fill missing features with 0
    df = df[feature_cols]

    # --- Predict RMSD (lower = better) ---
    preds = model.predict(df)
    df["predicted_rmsd"] = preds

    # --- Rank ascending (best = lowest RMSD) ---
    df["rank"] = df["predicted_rmsd"].rank(method="first", ascending=True).astype(int) - 1
    df = df.sort_values("rank")

    print(f"\nRanking for {prediction_dir.name}:")
    print(df[["predicted_rmsd", "rank"]])

    if return_dataframe:
        return df
    return df.index.tolist()