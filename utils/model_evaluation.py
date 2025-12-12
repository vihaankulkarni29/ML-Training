"""
Comprehensive model evaluation utilities.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss,
    matthews_corrcoef
)
from typing import Dict, Any, Optional


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
               - For binary: 1D array of positive class probabilities, or 2D array with shape (n_samples, 2)
               - For multi-class: 2D array with shape (n_samples, n_classes)
        class_names: Names of classes
        
    Returns:
        Dictionary containing metrics and figures:
        - classification_report: Precision, recall, F1 per class
        - confusion_matrix: Raw confusion matrix array
        - confusion_matrix_fig: Interactive Plotly heatmap
        - mcc: Matthews Correlation Coefficient
        - log_loss: Logarithmic loss (if y_proba provided)
        - roc_curve_fig: ROC curve (if binary and y_proba provided)
        - roc_auc: Area under ROC curve (if binary and y_proba provided)
    """
    results = {}
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    results['classification_report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm

    # Matthews Correlation Coefficient (robust for class imbalance)
    results['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix visualization
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
    ))
    
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=600
    )
    
    results['confusion_matrix_fig'] = fig_cm
    
    # Log Loss (if probabilities provided)
    if y_proba is not None:
        # For binary classification, y_proba should be 1D array of positive class probabilities
        # For multi-class, y_proba should be 2D array with shape (n_samples, n_classes)
        try:
            results['log_loss'] = log_loss(y_true, y_proba)
        except ValueError:
            # Handle case where y_proba format doesn't match expectations
            pass
    
    # ROC curve (for binary classification)
    if y_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='#2E86AB', width=2)
        ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig_roc.update_layout(
            title=f'ROC Curve (AUC = {roc_auc:.3f})',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=600,
            template='plotly_white'
        )
        
        results['roc_curve_fig'] = fig_roc
        results['roc_auc'] = roc_auc
    
    return results


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a regression model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing metrics and figures
    """
    results = {}
    
    # Calculate metrics
    results['mse'] = mean_squared_error(y_true, y_pred)
    results['rmse'] = np.sqrt(results['mse'])
    results['mae'] = mean_absolute_error(y_true, y_pred)
    results['r2'] = r2_score(y_true, y_pred)
    
    # Residuals
    residuals = y_true - y_pred
    results['residuals'] = residuals
    
    # Prediction vs Actual plot
    fig_pred = go.Figure()
    
    fig_pred.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='#2E86AB', size=8, opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig_pred.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_pred.update_layout(
        title=f'Predictions vs Actual (R² = {results["r2"]:.3f})',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        width=700,
        height=600,
        template='plotly_white'
    )
    
    results['prediction_fig'] = fig_pred
    
    # Residuals plot
    fig_residuals = go.Figure()
    
    fig_residuals.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(color='#A23B72', size=8, opacity=0.6)
    ))
    
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig_residuals.update_layout(
        title='Residual Plot',
        xaxis_title='Predicted Values',
        yaxis_title='Residuals',
        width=700,
        height=600,
        template='plotly_white'
    )
    
    results['residuals_fig'] = fig_residuals
    
    return results


def print_evaluation_summary(results: Dict[str, Any], task_type: str = "classification"):
    """
    Print formatted evaluation summary.
    
    Args:
        results: Results dictionary from evaluate_classifier or evaluate_regressor
        task_type: 'classification' or 'regression'
    """
    print("=" * 60)
    print(f"MODEL EVALUATION SUMMARY ({task_type.upper()})")
    print("=" * 60)
    
    if task_type == "classification":
        report = results['classification_report']
        print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)
        
        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"{label:<15} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} {metrics['f1-score']:<12.3f} {int(metrics['support']):<10}")
        
        print("-" * 60)
        print(f"{'Accuracy':<15} {report['accuracy']:.3f}")

        if 'mcc' in results:
            print(f"{'MCC':<15} {results['mcc']:.3f}")
        
        if 'roc_auc' in results:
            print(f"{'ROC AUC':<15} {results['roc_auc']:.3f}")
        
        if 'log_loss' in results:
            print(f"{'Log Loss':<15} {results['log_loss']:.3f}")
    
    elif task_type == "regression":
        print(f"\nR² Score:  {results['r2']:.4f}")
        print(f"RMSE:      {results['rmse']:.4f}")
        print(f"MAE:       {results['mae']:.4f}")
        print(f"MSE:       {results['mse']:.4f}")
    
    print("=" * 60)
