import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

def train_ranking(eval_csv, merged_csv, model_output="ranking_model.pkl"):
    """
    Train a regression model to rank Boltz predictions using combined_results.csv and
    merged_confidence_table.csv.
    """
    df_eval = pd.read_csv(eval_csv)
    df_features = pd.read_csv(merged_csv)

    # Create uppercase datapoint_id
    df_features["datapoint_id"] = (
        df_features["protein_id"].astype(str).str.upper() + "_" +
        df_features["binder_type"].str.upper() + "_" +
        df_features["ligand_id"].astype(str).str.upper()
    )
    df_eval["datapoint_id"] = df_eval["datapoint_id"].str.upper()

    # Melt evaluation table to get one RMSD per model
    df_eval_melt = df_eval.melt(
        id_vars=["datapoint_id"],
        value_vars=[f"rmsd_model_{i}" for i in range(5)],
        var_name="model_idx_str",
        value_name="rmsd"
    )
    df_eval_melt["model_idx"] = df_eval_melt["model_idx_str"].str.extract(r"rmsd_model_(\d)").astype(int)
    df_eval_melt = df_eval_melt.drop(columns="model_idx_str")

    # Align column names
    if "model_id" in df_features.columns:
        df_features = df_features.rename(columns={"model_id": "model_idx"})

    # Merge datasets
    df_train = pd.merge(df_features, df_eval_melt, how="inner", on=["datapoint_id", "model_idx"])
    print(f"✅ Training samples after merge: {len(df_train)}")

    # Identify features (drop irrelevant columns)
    exclude_cols = [
        "datapoint_id", "protein_id", "ligand_id", "binder_type",
        "model_idx", "rmsd", "type"
    ]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Keep only numeric columns
    non_numeric_cols = df_train[feature_cols].select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        print(f"⚠️ Dropping non-numeric columns: {non_numeric_cols}")
        feature_cols = [c for c in feature_cols if c not in non_numeric_cols]

    X = df_train[feature_cols]
    print(X.columns)
    y = df_train["rmsd"]

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Save model and features
    joblib.dump((model, feature_cols), model_output)
    print(f"✅ Model trained and saved to {model_output}")

    return model, df_train



if __name__ == "__main__":
    eval_csv_path = "combined_results.csv"
    merged_csv_path = "merged_confidence_table.csv"
    train_ranking(eval_csv_path, merged_csv_path)