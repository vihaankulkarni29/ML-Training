"""
Reusable visualization utilities for consistent Plotly charts across projects.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List


# Consistent color schemes
COLOR_SCHEMES = {
    "default": px.colors.qualitative.Set2,
    "professional": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"],
    "gradient": px.colors.sequential.Viridis,
}


def create_distribution_plot(
    df: pd.DataFrame,
    column: str,
    color_by: Optional[str] = None,
    title: Optional[str] = None,
    nbins: int = 30
) -> go.Figure:
    """
    Create an interactive histogram with KDE overlay.
    
    Args:
        df: Input dataframe
        column: Column to plot
        color_by: Column to color by
        title: Plot title
        nbins: Number of bins
        
    Returns:
        Plotly figure
    """
    if title is None:
        title = f"Distribution of {column}"
    
    fig = px.histogram(
        df,
        x=column,
        color=color_by,
        nbins=nbins,
        title=title,
        marginal="box",  # Add box plot on top
        hover_data=df.columns,
        color_discrete_sequence=COLOR_SCHEMES["professional"]
    )
    
    fig.update_layout(
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Matrix"
) -> go.Figure:
    """
    Create an interactive correlation heatmap.
    
    Args:
        df: Input dataframe
        columns: Columns to include (default: all numeric)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if columns is None:
        df_corr = df.select_dtypes(include=[np.number]).corr()
    else:
        df_corr = df[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=df_corr.values,
        x=df_corr.columns,
        y=df_corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=df_corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        width=800,
        height=800,
        template='plotly_white'
    )
    
    return fig


def create_scatter_with_trend(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: Optional[str] = None,
    title: Optional[str] = None,
    trendline: str = "ols"
) -> go.Figure:
    """
    Create scatter plot with trendline.
    
    Args:
        df: Input dataframe
        x: X-axis column
        y: Y-axis column
        color_by: Column to color by
        title: Plot title
        trendline: Type of trendline ('ols', 'lowess', etc.)
        
    Returns:
        Plotly figure
    """
    if title is None:
        title = f"{y} vs {x}"
    
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color_by,
        title=title,
        trendline=trendline,
        hover_data=df.columns,
        color_discrete_sequence=COLOR_SCHEMES["professional"]
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(template='plotly_white')
    
    return fig


def create_grouped_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create grouped bar chart.
    
    Args:
        df: Input dataframe
        x: X-axis column (categorical)
        y: Y-axis column (numeric)
        color: Grouping column
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if title is None:
        title = f"{y} by {x} (grouped by {color})"
    
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        barmode='group',
        title=title,
        color_discrete_sequence=COLOR_SCHEMES["professional"]
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_time_series_plot(
    df: pd.DataFrame,
    date_column: str,
    value_columns: List[str],
    title: str = "Time Series Analysis"
) -> go.Figure:
    """
    Create interactive time series plot.
    
    Args:
        df: Input dataframe
        date_column: Column with datetime values
        value_columns: List of columns to plot
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for i, col in enumerate(value_columns):
        fig.add_trace(go.Scatter(
            x=df[date_column],
            y=df[col],
            mode='lines+markers',
            name=col,
            line=dict(color=COLOR_SCHEMES["professional"][i % len(COLOR_SCHEMES["professional"])])
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Add range slider
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig


def save_figure(fig: go.Figure, filepath: str, include_html: bool = True):
    """
    Save Plotly figure as static image and optionally HTML.
    
    Args:
        fig: Plotly figure
        filepath: Output path (without extension)
        include_html: Whether to save interactive HTML version
    """
    from pathlib import Path
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as PNG
    fig.write_image(str(filepath.with_suffix('.png')), width=1200, height=800)
    
    # Save as HTML
    if include_html:
        fig.write_html(str(filepath.with_suffix('.html')))
    
    print(f"📊 Figure saved to {filepath}")
