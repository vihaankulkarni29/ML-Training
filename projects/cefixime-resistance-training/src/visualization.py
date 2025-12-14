"""
Visualization utilities using Plotly.
"""

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Create an interactive confusion matrix using Plotly.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (optional)
        
    Returns:
        Plotly figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f"Class {i}" for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=600
    )
    
    return fig


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance from a trained model.
    
    Args:
        model: Trained sklearn model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to display
        
    Returns:
        Plotly figure
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model doesn't have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'Importance': 'Feature Importance', 'Feature': ''},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig
