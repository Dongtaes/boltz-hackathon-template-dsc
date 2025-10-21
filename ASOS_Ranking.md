# Final Submission: Boltz Confidence-Based Ranking Model

## Problem Overview
Each molecule target in the dataset has five predicted structures (`model_0`–`model_4`), each accompanied by a `confidence_*.json` file containing Boltz confidence metrics such as:

- `confidence_score`
- `ligand_iptm`
- `complex_iplddt`
- `complex_pde`

Due to a bias toward orthosteric binders in the training dataset, predictions for allosteric binders are often less accurate. A supervised ranking approach was developed to reliably select the best prediction for each target.


## Model Training
An **XGBoost regressor** was trained using:

- **Input features:** numeric Boltz confidence metrics.
- **Target:** per-model RMSD (lower RMSD indicates a better prediction).
- **Output:** predicted ranking score.

The model learns to map confidence metrics to prediction quality, enabling robust ranking for new targets.

## Prediction and Ranking
For each new target folder, the function `rank_asos_predictions()`:

1. Loads all five `confidence_*.json` files.  
2. Extracts and normalizes relevant features.  
3. Uses the trained XGBoost model to predict a ranking score for each model.  
4. Returns an ordered list of model IDs from best to worst.

The top-ranked model (rank 0) is then used for evaluation.

### Example Usage
```python
from rank_asos import rank_asos_predictions

ranking = rank_asos_predictions(
    "/path/to/predictions",
    model_path="trained_ranking_model.pkl"
)
print("Model ranking:", ranking)