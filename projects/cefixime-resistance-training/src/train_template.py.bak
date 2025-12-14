"""
Model training utilities.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path


def create_baseline_pipeline(estimator):
    """
    Create a baseline sklearn pipeline.
    
    Args:
        estimator: Sklearn estimator (e.g., LogisticRegression())
        
    Returns:
        Configured pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', estimator)
    ])


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    """
    Train pipeline and print evaluation metrics.
    
    Args:
        pipeline: Sklearn pipeline
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Trained pipeline
    """
    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)
    
    print("✅ Training complete!")
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"\nTrain Accuracy: {train_score:.4f}")
    print(f"Test Accuracy:  {test_score:.4f}")
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    return pipeline


def save_model(pipeline, filepath: str = "models/model.pkl"):
    """
    Save trained model to disk.
    
    Args:
        pipeline: Trained pipeline
        filepath: Output path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"💾 Model saved to {filepath}")
