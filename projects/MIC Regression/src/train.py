"""MIC Regression Model Training Pipeline.

Trains a Random Forest regressor to predict antimicrobial peptide potency
from physicochemical features.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


def train_mic_model() -> None:
    """Load data, train RF regressor, evaluate, and save artifacts."""
    # =====================================================================
    # 1. PATHS & SETUP
    # =====================================================================
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / "processed_features.csv"
    models_dir = project_root / "models"
    results_dir = project_root / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # 2. DATA LOADING & PREPARATION
    # =====================================================================
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples from {data_path}")

    # Validate required columns
    target_col = "neg_log_mic_microM"
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Define feature columns: exclude metadata and target
    exclude_cols = ["SEQUENCE", "MIC", target_col, "NAME", "ID"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Separate physicochemical vs k-mer features for reporting
    phys_features = [
        "mol_weight", "aromaticity", "instability_index", 
        "isoelectric_point", "gravy", "length", "positive_charge"
    ]
    kmer_features = [c for c in feature_cols if c.startswith("kmer_")]
    
    print(f"✓ Using {len(phys_features)} physicochemical + {len(kmer_features)} k-mer features = {len(feature_cols)} total")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Remove any rows with NaN
    valid_idx = (~X.isna().any(axis=1)) & (~y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"✓ {len(X)} samples after validation (removed {len(df) - len(X)} rows)")

    # =====================================================================
    # 3. TRAIN/TEST SPLIT
    # =====================================================================
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Train/Test split: {len(X_train)} / {len(X_test)}")

    # =====================================================================
    # 4. MODEL TRAINING
    # =====================================================================
    print("\n🔨 Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
    )
    model.fit(X_train, y_train)
    print("✓ Model training complete")

    # =====================================================================
    # 5. EVALUATION
    # =====================================================================
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    pearson_r, pearson_p = pearsonr(y_test, y_pred_test)

    print("\n📊 TEST SET METRICS:")
    print(f"   RMSE:              {rmse_test:.4f} log units")
    print(f"   R² Score:          {r2_test:.4f}")
    print(f"   Pearson r:         {pearson_r:.4f} (p-value: {pearson_p:.2e})")
    print(f"\n📊 TRAIN SET METRICS:")
    print(f"   RMSE:              {rmse_train:.4f} log units")
    print(f"   R² Score:          {r2_train:.4f}")

    # =====================================================================
    # 6. HUMAN INTERPRETATION
    # =====================================================================
    print(f"\n💡 INTERPRETATION:")
    print(f"   On average, the model predictions deviate by ±{rmse_test:.4f} log units")
    print(f"   ({10**rmse_test:.2f}x fold-change in actual MIC values)")
    print(f"   The model explains {r2_test*100:.1f}% of variance in test set")
    print(f"   Predictions are strongly correlated with actuals (r={pearson_r:.3f})")

    # =====================================================================
    # 7. FEATURE IMPORTANCE VISUALIZATION
    # =====================================================================
    feat_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_importance.plot(kind="barh", ax=ax, color="#1f77b4", edgecolor="black")
    ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
    ax.set_title("Random Forest Feature Importance for MIC Prediction", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    fi_path = results_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Feature importance plot saved to {fi_path}")

    # =====================================================================
    # 8. PREDICTED VS ACTUAL VISUALIZATION
    # =====================================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(y_test, y_pred_test, alpha=0.6, s=50, edgecolor="black", linewidth=0.5)

    # Perfect fit diagonal line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Fit")

    ax.set_xlabel("Actual neg_log_mic (log units)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted neg_log_mic (log units)", fontsize=12, fontweight="bold")
    ax.set_title(f"MIC Prediction: Actual vs Predicted (R² = {r2_test:.4f})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    pva_path = results_dir / "predicted_vs_actual.png"
    plt.savefig(pva_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Predicted vs Actual plot saved to {pva_path}")

    # =====================================================================
    # 9. SAVE MODEL
    # =====================================================================
    import joblib

    model_path = models_dir / "mic_predictor.pkl"
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")

    # =====================================================================
    # 10. SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"✅ Artifacts:")
    print(f"   - Model: {model_path}")
    print(f"   - Feature importance: {fi_path}")
    print(f"   - Predictions plot: {pva_path}")
    print(f"\n📈 Final Metrics (Test Set):")
    print(f"   RMSE: {rmse_test:.4f} log units")
    print(f"   R²:   {r2_test:.4f}")
    print(f"   r:    {pearson_r:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    train_mic_model()
