"""
Model training utilities with medical AI focus.
Prioritizes sensitivity (recall) for antibiotic resistance prediction.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import plotly.graph_objects as go
import plotly.express as px


def calculate_specificity(y_true, y_pred):
    """
    Calculate specificity (True Negative Rate).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Specificity score
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return specificity


def plot_confusion_matrix_medical(y_true, y_pred, output_path):
    """
    Create medical-focused confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Medical terminology labels
    labels = ['Susceptible (0)', 'Resistant (1)']
    
    # Create annotations with medical context
    annotations = [
        [f'TN: {tn}<br>Correctly ID\'d<br>Susceptible', f'FP: {fp}<br>False Alarm<br>(Low Risk)'],
        [f'FN: {fn}<br>⚠️ MISSED<br>Resistant', f'TP: {tp}<br>Correctly ID\'d<br>Resistant']
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='RdYlGn_r',  # Red for errors, green for correct
        text=annotations,
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>',
        showscale=False
    ))
    
    fig.update_layout(
        title={
            'text': 'Confusion Matrix - Ceftriaxone Resistance<br><sub>⚠️ Focus: Minimize False Negatives (FN)</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=700,
        height=700,
        font=dict(size=14),
        template='plotly_white'
    )
    
    # Save figure
    fig.write_image(str(output_path), width=700, height=700)
    print(f"   ✓ Confusion matrix saved to {output_path}")


def train_resistance_model(data_path: str, output_dir: str = "results"):
    """
    Train antibiotic resistance prediction model with medical AI best practices.
    
    Args:
        data_path: Path to processed dataset CSV
        output_dir: Directory to save results
    """
    print("=" * 70)
    print("ANTIBIOTIC RESISTANCE MODEL TRAINING - MEDICAL AI PIPELINE")
    print("=" * 70)
    
    # Create output directories
    results_path = Path(output_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    models_path = Path("models")
    models_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("\n📂 Loading dataset...")
    df = pd.read_csv(data_path, index_col=0)  # BioSample is index
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {df.shape[1] - 1} AMR genes")
    print(f"   Samples: {df.shape[0]}")
    
    # Separate features and target
    X = df.drop(columns=['resistance_label'])
    y = df['resistance_label']
    
    # Class distribution
    resistance_rate = y.mean()
    print(f"\n📊 Class Distribution:")
    print(f"   Susceptible (0): {(1-resistance_rate)*100:.1f}%")
    print(f"   Resistant (1):   {resistance_rate*100:.1f}%")
    print(f"   ⚠️  Imbalanced dataset - using class_weight='balanced'")
    
    # 2. Stratified Train-Test Split
    print("\n🔀 Performing stratified 80/20 train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y,  # Maintain class balance
        random_state=42
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # 3. Train Random Forest Model
    print("\n🌲 Training Random Forest Classifier...")
    print("   Hyperparameters:")
    print("   - n_estimators: 100")
    print("   - class_weight: 'balanced' (to handle imbalance)")
    print("   - random_state: 42")
    print("   - n_jobs: -1 (parallel processing)")
    
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',  # Critical for medical AI
        random_state=42,
        n_jobs=-1,
        max_depth=10,  # Prevent overfitting
        min_samples_split=5
    )
    
    model.fit(X_train, y_train)
    print("   ✅ Model training complete!")
    
    # 4. Make Predictions
    print("\n🔮 Generating predictions on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of resistance
    
    # 5. Calculate Medical AI Metrics
    print("\n" + "=" * 70)
    print("MEDICAL AI EVALUATION METRICS")
    print("=" * 70)
    
    # Core metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)  # Recall = Sensitivity
    specificity = calculate_specificity(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Display metrics with medical context
    print(f"\n{'Metric':<25} {'Value':<10} {'Medical Interpretation'}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {accuracy:.3f}      Overall correctness")
    print(f"{'Sensitivity (Recall)':<25} {sensitivity:.3f}      ⚠️  CRITICAL: Catch resistant cases")
    print(f"{'Specificity':<25} {specificity:.3f}      Correctly ID susceptible cases")
    print(f"{'ROC-AUC Score':<25} {roc_auc:.3f}      Model discrimination ability")
    
    # Clinical interpretation
    print("\n" + "=" * 70)
    print("CLINICAL INTERPRETATION")
    print("=" * 70)
    
    if sensitivity >= 0.90:
        sens_status = "✅ EXCELLENT"
    elif sensitivity >= 0.80:
        sens_status = "⚠️  ACCEPTABLE"
    else:
        sens_status = "❌ NEEDS IMPROVEMENT"
    
    print(f"\nSensitivity Status: {sens_status}")
    print(f"   → We correctly identify {sensitivity*100:.1f}% of resistant cases")
    print(f"   → We miss {(1-sensitivity)*100:.1f}% of resistant cases (False Negatives)")
    
    if specificity >= 0.85:
        spec_status = "✅ GOOD"
    else:
        spec_status = "⚠️  Some false alarms"
    
    print(f"\nSpecificity Status: {spec_status}")
    print(f"   → We correctly identify {specificity*100:.1f}% of susceptible cases")
    print(f"   → {(1-specificity)*100:.1f}% false alarms (False Positives)")
    
    # 6. Detailed Classification Report
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=['Susceptible', 'Resistant']))
    
    # 7. Generate Confusion Matrix Visualization
    print("\n📊 Generating confusion matrix visualization...")
    cm_path = results_path / "confusion_matrix.png"
    plot_confusion_matrix_medical(y_test, y_pred, cm_path)
    
    # 8. Extract and Save Feature Importance
    print("\n🧬 Analyzing gene importance...")
    feature_importance = pd.DataFrame({
        'gene': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save top 20
    top20_path = results_path / "feature_importance.csv"
    feature_importance.head(20).to_csv(top20_path, index=False)
    print(f"   ✓ Top 20 features saved to {top20_path}")
    
    # Display top 10
    print("\n🔬 Top 10 Most Important AMR Genes:")
    print("-" * 50)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['gene']:<30} {row['importance']:.4f}")
    
    # 9. Save Model
    print("\n💾 Saving trained model...")
    model_path = models_path / "ceftriaxone_model.pkl"
    joblib.dump(model, model_path)
    print(f"   ✓ Model saved to {model_path}")
    
    # 10. Summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n✅ Model Artifacts:")
    print(f"   - Trained model: {model_path}")
    print(f"   - Confusion matrix: {cm_path}")
    print(f"   - Feature importance: {top20_path}")
    
    print(f"\n📈 Key Takeaway:")
    print(f"   Sensitivity: {sensitivity:.1%} - We catch {sensitivity:.1%} of resistant cases")
    print(f"   Specificity: {specificity:.1%} - We correctly ID {specificity:.1%} of susceptible cases")
    
    if sensitivity < 0.85:
        print(f"\n⚠️  WARNING: Sensitivity below 85% - Consider:")
        print(f"   1. Adjusting decision threshold (lower = more sensitive)")
        print(f"   2. Feature engineering (gene interactions)")
        print(f"   3. Ensemble methods or boosting")
    
    print("\n" + "=" * 70)
    
    return model, feature_importance


if __name__ == "__main__":
    # Run training pipeline
    DATA_PATH = "data/processed/dataset_ceftriaxone.csv"
    
    print("\n🚀 Starting Ceftriaxone Resistance Model Training...")
    print("   Medical AI Focus: Maximize Sensitivity (catch all resistant cases)\n")
    
    model, importance = train_resistance_model(DATA_PATH)
    
    print("\n✨ Training complete! Check the results/ folder for artifacts.")
