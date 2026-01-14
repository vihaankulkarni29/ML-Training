"""
1D Grad-CAM (Gradient-weighted Class Activation Mapping) for DeepG2P Model
Explainable AI for antimicrobial resistance prediction from mass spectrometry signals
Includes both static (matplotlib) and interactive (plotly) visualizations
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import DeepG2P, create_deepg2p_model


class GradCAM1D:
    """
    1D Gradient-weighted Class Activation Mapping for explainable deep learning.
    
    Visualizes which regions of 1D input signals are important for model predictions.
    
    Args:
        model: Trained PyTorch model
        target_layer_name: Name of layer to visualize (e.g., 'layer4')
        device: torch device
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer_name: str = 'layer4',
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.model.eval()
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        
        # Storage for forward/backward hooks
        self.activations = None
        self.gradients = None
        
        # Find target layer
        self.target_layer = self._find_layer(target_layer_name)
        if self.target_layer is None:
            raise ValueError(f"Layer '{target_layer_name}' not found in model")
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._backward_hook)
        
        print(f"✓ GradCAM initialized on layer: {target_layer_name}")
        print(f"✓ Device: {self.device}")
    
    def _find_layer(self, layer_name: str):
        """Find layer by name in model."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def _forward_hook(self, module, input, output):
        """Capture activations during forward pass."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass."""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(
        self,
        input_signal: torch.Tensor,
        target_class: int,
        return_prediction: bool = True
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Generate Grad-CAM heatmap for input signal.
        
        Args:
            input_signal: Input tensor [1, channels, length] or [channels, length]
            target_class: Target class index for explanation
            return_prediction: Whether to return model prediction
            
        Returns:
            heatmap: 1D numpy array normalized to [0, 1]
            prediction: Model's predicted probability (optional)
        """
        # Ensure correct shape [1, C, L]
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)
        elif input_signal.dim() == 1:
            input_signal = input_signal.unsqueeze(0).unsqueeze(0)
        
        input_signal = input_signal.to(self.device)
        input_signal.requires_grad = True
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_signal)
        
        # Get prediction for target class
        target_score = output[0, target_class]
        prediction = torch.sigmoid(target_score).item() if return_prediction else None
        
        # Backward pass
        target_score.backward()
        
        # Compute Grad-CAM
        # 1. Get activations and gradients
        activations = self.activations  # [1, C, L']
        gradients = self.gradients      # [1, C, L']
        
        # 2. Global average pooling of gradients (weights)
        weights = torch.mean(gradients, dim=2, keepdim=True)  # [1, C, 1]
        
        # 3. Weighted combination of activation maps
        weighted_activations = activations * weights  # [1, C, L']
        heatmap = torch.sum(weighted_activations, dim=1, keepdim=True)  # [1, 1, L']
        
        # 4. ReLU (only positive contributions)
        heatmap = F.relu(heatmap)
        
        # 5. Upsample to original signal length
        original_length = input_signal.shape[2]
        heatmap = F.interpolate(
            heatmap,
            size=original_length,
            mode='linear',
            align_corners=False
        )
        
        # 6. Normalize to [0, 1]
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap, prediction
    
    def cleanup(self):
        """Remove hooks to free memory."""
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass


def plot_interactive_gradcam(
    signal: np.ndarray,
    heatmap: np.ndarray,
    prediction: float,
    target_class: int,
    antibiotic_name: str = "Antibiotic",
    top_regions: List[Tuple[int, int, float]] = None,
    output_path: Optional[str] = None
):
    """
    Create interactive Plotly visualization of Grad-CAM results.
    
    Args:
        signal: Original 1D signal
        heatmap: Grad-CAM heatmap
        prediction: Model prediction probability
        target_class: Target class index
        antibiotic_name: Name of antibiotic
        top_regions: List of (start, end, importance) tuples
        output_path: Path to save HTML file
    
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Mass Spectrum Signal (6000 m/z bins)',
            f'Grad-CAM Heatmap - Class {target_class} ({antibiotic_name})',
            'Signal with Grad-CAM Overlay (Red = High Importance)'
        ),
        vertical_spacing=0.12,
        row_heights=[0.33, 0.33, 0.34]
    )
    
    x_axis = np.arange(len(signal))
    
    # Row 1: Original Signal
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=signal,
            mode='lines',
            name='Mass Spectrum',
            line=dict(color='#2E86AB', width=1.5),
            hovertemplate='m/z Index: %{x}<br>Intensity: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add statistics box
    stats_text = f"Mean: {signal.mean():.3f}<br>Std: {signal.std():.3f}<br>Max: {signal.max():.3f}"
    fig.add_annotation(
        text=stats_text,
        xref="x1", yref="y1",
        x=len(signal) * 0.98, y=signal.max() * 0.95,
        showarrow=False,
        bgcolor="wheat",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
        align="right",
        row=1, col=1
    )
    
    # Row 2: Grad-CAM Heatmap
    # Create color array for heatmap
    colors = plt.cm.hot(heatmap)
    
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=heatmap,
            mode='lines',
            name='Grad-CAM',
            fill='tozeroy',
            line=dict(color='darkred', width=2),
            fillcolor='rgba(255, 0, 0, 0.3)',
            hovertemplate='Position: %{x}<br>Attribution: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add colorscale indicator
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=heatmap,
            mode='markers',
            marker=dict(
                size=0.1,
                color=heatmap,
                colorscale='Hot',
                showscale=True,
                colorbar=dict(
                    title="Importance",
                    len=0.3,
                    y=0.5,
                    yanchor="middle"
                )
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # Row 3: Signal with Overlay
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=signal,
            mode='lines',
            name='Signal',
            line=dict(color='#2E86AB', width=1.5),
            opacity=0.7,
            hovertemplate='m/z Index: %{x}<br>Intensity: %{y:.4f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Add heatmap overlay as filled regions
    # Find contiguous high-importance regions
    threshold = 0.3
    high_importance = heatmap > threshold
    
    in_region = False
    start_idx = 0
    
    for i, is_important in enumerate(high_importance):
        if is_important and not in_region:
            start_idx = i
            in_region = True
        elif not is_important and in_region:
            # Add shaded region
            avg_importance = heatmap[start_idx:i].mean()
            fig.add_vrect(
                x0=start_idx, x1=i,
                fillcolor="red",
                opacity=avg_importance * 0.4,
                layer="below",
                line_width=0,
                row=3, col=1
            )
            in_region = False
    
    # Mark top regions if provided
    if top_regions:
        for idx, (start, end, importance) in enumerate(top_regions[:3], 1):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="orange",
                opacity=0.2,
                layer="above",
                line=dict(color="orange", width=2, dash="dash"),
                annotation_text=f"Top {idx}",
                annotation_position="top left",
                row=3, col=1
            )
    
    # Update layout
    resistance_status = "RESISTANT" if prediction > 0.5 else "SUSCEPTIBLE"
    confidence = prediction if prediction > 0.5 else (1 - prediction)
    
    fig.update_layout(
        title=dict(
            text=f"<b>Explainable AI: {antibiotic_name} Resistance Prediction</b><br>" +
                 f"<sub>Prediction: {resistance_status} (Confidence: {confidence:.1%})</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        height=1000,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    # Update axes
    fig.update_xaxes(title_text="m/z Index", row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Signal Position", row=2, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="m/z Index", row=3, col=1, gridcolor='lightgray')
    
    fig.update_yaxes(title_text="Intensity", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Attribution Score", row=2, col=1, range=[0, 1], gridcolor='lightgray')
    fig.update_yaxes(title_text="Intensity", row=3, col=1, gridcolor='lightgray')
    
    # Save interactive HTML
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✓ Interactive figure saved to {output_path}")
    
    return fig


def plot_static_gradcam(
    signal: np.ndarray,
    heatmap: np.ndarray,
    prediction: float,
    target_class: int,
    antibiotic_name: str = "Antibiotic",
    output_path: Optional[str] = None,
    dpi: int = 300
):
    """
    Create static matplotlib visualization of Grad-CAM results.
    
    Args:
        signal: Original 1D signal
        heatmap: Grad-CAM heatmap
        prediction: Model prediction probability
        target_class: Target class index
        antibiotic_name: Name of antibiotic
        output_path: Path to save figure
        dpi: Resolution for saved figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Color schemes
    signal_color = '#2E86AB'  # Blue
    heatmap_cmap = plt.cm.hot
    
    # Axis 0: Original Signal
    axes[0].plot(signal, color=signal_color, linewidth=1.5, alpha=0.8)
    axes[0].set_title(
        f'Mass Spectrum Signal (6000 m/z bins)',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    axes[0].set_xlabel('m/z Index', fontsize=12)
    axes[0].set_ylabel('Intensity', fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim(0, len(signal))
    
    # Add statistics box
    textstr = f'Mean: {signal.mean():.3f}\nStd: {signal.std():.3f}\nMax: {signal.max():.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0].text(
        0.98, 0.97, textstr,
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
    )
    
    # Axis 1: Grad-CAM Heatmap
    x = np.arange(len(heatmap))
    colors = heatmap_cmap(heatmap)
    
    for i in range(len(x) - 1):
        axes[1].fill_between(
            [x[i], x[i+1]],
            0, heatmap[i],
            color=colors[i],
            alpha=0.8
        )
    
    axes[1].plot(heatmap, color='darkred', linewidth=2, alpha=0.9, label='Activation Intensity')
    axes[1].set_title(
        f'Grad-CAM Heatmap - Class {target_class} ({antibiotic_name})',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    axes[1].set_xlabel('Signal Position', fontsize=12)
    axes[1].set_ylabel('Attribution Score', fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlim(0, len(heatmap))
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='upper right')
    
    # Add colorbar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], orientation='vertical', pad=0.01)
    cbar.set_label('Importance', fontsize=10)
    
    # Axis 2: Signal with Overlay
    axes[2].plot(signal, color=signal_color, linewidth=1.5, alpha=0.6, label='Mass Spectrum')
    
    # Overlay heatmap as colored background
    for i in range(len(x) - 1):
        if heatmap[i] > 0.3:  # Only show strong attributions
            axes[2].axvspan(
                x[i], x[i+1],
                alpha=heatmap[i] * 0.5,
                color='red',
                linewidth=0
            )
    
    axes[2].set_title(
        f'Signal with Grad-CAM Overlay (Red = High Importance)',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    axes[2].set_xlabel('m/z Index', fontsize=12)
    axes[2].set_ylabel('Intensity', fontsize=12)
    axes[2].set_xlim(0, len(signal))
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(loc='upper right')
    
    # Add prediction information
    resistance_status = "RESISTANT" if prediction > 0.5 else "SUSCEPTIBLE"
    confidence = prediction if prediction > 0.5 else (1 - prediction)
    
    fig.suptitle(
        f'Explainable AI: {antibiotic_name} Resistance Prediction\n'
        f'Prediction: {resistance_status} (Confidence: {confidence:.1%})',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Static figure saved to {output_path}")
    
    return fig


def find_top_regions(heatmap: np.ndarray, top_k: int = 5) -> List[Tuple[int, int, float]]:
    """
    Find top-k most important regions in heatmap.
    
    Args:
        heatmap: Grad-CAM heatmap
        top_k: Number of top regions to return
        
    Returns:
        List of (start_idx, end_idx, importance_score) tuples
    """
    # Threshold heatmap
    threshold = np.percentile(heatmap, 90)
    important_mask = heatmap > threshold
    
    # Find contiguous regions
    regions = []
    in_region = False
    start_idx = 0
    
    for i, is_important in enumerate(important_mask):
        if is_important and not in_region:
            start_idx = i
            in_region = True
        elif not is_important and in_region:
            importance = heatmap[start_idx:i].mean()
            regions.append((start_idx, i, importance))
            in_region = False
    
    # Handle case where region extends to end
    if in_region:
        importance = heatmap[start_idx:].mean()
        regions.append((start_idx, len(heatmap), importance))
    
    # Sort by importance and return top-k
    regions.sort(key=lambda x: x[2], reverse=True)
    return regions[:top_k]


def explain_sample(
    model_path: str,
    data_path: str,
    sample_idx: Optional[int] = None,
    target_class: int = 0,
    antibiotic_names: List[str] = None,
    output_dir: str = 'results/explanations',
    layer_name: str = 'layer4',
    interactive: bool = True
):
    """
    Generate Grad-CAM explanation for a specific sample.
    
    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to data .npy file
        sample_idx: Index of sample to explain (None = random resistant sample)
        target_class: Class index to explain
        antibiotic_names: Names of antibiotics
        output_dir: Directory to save results
        layer_name: Target layer for Grad-CAM
        interactive: Whether to generate interactive Plotly visualization
    """
    print("=" * 70)
    print("GRAD-CAM EXPLAINABLE AI - ANTIMICROBIAL RESISTANCE")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n📂 Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Infer model configuration from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer num_antibiotics from checkpoint
    if 'fc.weight' in state_dict:
        num_antibiotics_from_checkpoint = state_dict['fc.weight'].shape[0]
        print(f"   Detected {num_antibiotics_from_checkpoint} antibiotics from checkpoint")
    else:
        num_antibiotics_from_checkpoint = 10
        print(f"   Using default {num_antibiotics_from_checkpoint} antibiotics")
    
    # Create model
    model = create_deepg2p_model(
        input_length=6000,
        input_channels=1,
        num_antibiotics=num_antibiotics_from_checkpoint,
        model_size='medium'
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"✓ Model loaded successfully")
    
    # Load data
    print(f"\n📊 Loading data from {data_path}...")
    X = np.load(data_path)
    print(f"✓ Data shape: {X.shape}")
    
    # Load labels if available
    labels_path = data_path.replace('X', 'y')
    y = None
    if Path(labels_path).exists():
        y = np.load(labels_path)
        print(f"✓ Labels shape: {y.shape}")
    
    # Select sample
    if sample_idx is None and y is not None:
        # Find resistant samples for target class
        resistant_samples = np.where(y[:, target_class] == 1)[0]
        if len(resistant_samples) > 0:
            sample_idx = np.random.choice(resistant_samples)
            print(f"\n🎯 Selected random resistant sample: {sample_idx}")
        else:
            sample_idx = np.random.randint(0, len(X))
            print(f"\n🎯 No resistant samples found, selected random sample: {sample_idx}")
    elif sample_idx is None:
        sample_idx = np.random.randint(0, len(X))
        print(f"\n🎯 Selected random sample: {sample_idx}")
    
    signal = X[sample_idx]
    
    # Initialize Grad-CAM
    print(f"\n🔍 Initializing Grad-CAM on layer '{layer_name}'...")
    gradcam = GradCAM1D(model, target_layer_name=layer_name, device=device)
    
    # Generate heatmap
    print(f"\n🔥 Generating Grad-CAM heatmap for class {target_class}...")
    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
    heatmap, prediction = gradcam.generate_heatmap(
        signal_tensor,
        target_class=target_class,
        return_prediction=True
    )
    
    print(f"✓ Heatmap generated")
    print(f"✓ Model prediction: {prediction:.4f} ({'RESISTANT' if prediction > 0.5 else 'SUSCEPTIBLE'})")
    
    # Find top important regions
    print(f"\n📍 Top 5 Important Regions:")
    top_regions = find_top_regions(heatmap, top_k=5)
    for i, (start, end, importance) in enumerate(top_regions, 1):
        print(f"   {i}. m/z [{start:4d} - {end:4d}]: Importance = {importance:.3f}")
    
    # Create visualizations
    antibiotic_name = antibiotic_names[target_class] if antibiotic_names else f"Class {target_class}"
    
    # Interactive visualization
    if interactive:
        print(f"\n📊 Creating interactive visualization...")
        interactive_file = output_path / f'gradcam_interactive_sample_{sample_idx}_class_{target_class}.html'
        
        fig_interactive = plot_interactive_gradcam(
            signal=signal,
            heatmap=heatmap,
            prediction=prediction,
            target_class=target_class,
            antibiotic_name=antibiotic_name,
            top_regions=top_regions,
            output_path=str(interactive_file)
        )
    
    # Static visualization
    print(f"\n📊 Creating static visualization...")
    static_file = output_path / f'gradcam_static_sample_{sample_idx}_class_{target_class}.png'
    
    fig_static = plot_static_gradcam(
        signal=signal,
        heatmap=heatmap,
        prediction=prediction,
        target_class=target_class,
        antibiotic_name=antibiotic_name,
        output_path=str(static_file),
        dpi=300
    )
    
    # Save heatmap data
    np.save(output_path / f'heatmap_sample_{sample_idx}_class_{target_class}.npy', heatmap)
    
    # Save interpretation report
    report_path = output_path / f'report_sample_{sample_idx}_class_{target_class}.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GRAD-CAM EXPLANATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Sample Index: {sample_idx}\n")
        f.write(f"Target Class: {target_class} ({antibiotic_name})\n")
        f.write(f"Prediction: {prediction:.4f} ({'RESISTANT' if prediction > 0.5 else 'SUSCEPTIBLE'})\n")
        f.write(f"Confidence: {max(prediction, 1-prediction):.1%}\n\n")
        f.write("Top 5 Important Regions:\n")
        f.write("-" * 70 + "\n")
        for i, (start, end, importance) in enumerate(top_regions, 1):
            f.write(f"{i}. m/z [{start:4d} - {end:4d}]: Importance = {importance:.3f}\n")
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ Report saved to {report_path}")
    
    # Cleanup
    gradcam.cleanup()
    
    print("\n" + "=" * 70)
    print("EXPLANATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 Outputs:")
    if interactive:
        print(f"   - Interactive visualization: {interactive_file}")
    print(f"   - Static visualization: {static_file}")
    print(f"   - Heatmap data: {output_path / f'heatmap_sample_{sample_idx}_class_{target_class}.npy'}")
    print(f"   - Report: {report_path}")
    print("=" * 70)
    
    return heatmap, prediction


def main():
    """Main entry point for Grad-CAM explanation."""
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM explanations for DeepG2P model'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/best_model.pth',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/X.npy',
        help='Path to input data .npy file'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=None,
        help='Sample index to explain (None = random resistant sample)'
    )
    parser.add_argument(
        '--target-class',
        type=int,
        default=0,
        help='Target class index to explain (default: 0)'
    )
    parser.add_argument(
        '--antibiotics',
        type=str,
        nargs='+',
        default=['Ciprofloxacin', 'Ceftriaxone', 'Meropenem'],
        help='Names of antibiotics'
    )
    parser.add_argument(
        '--layer',
        type=str,
        default='layer4',
        help='Target layer for Grad-CAM (default: layer4)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/explanations',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive Plotly visualization'
    )
    
    args = parser.parse_args()
    
    # Run explanation
    explain_sample(
        model_path=args.model_path,
        data_path=args.data_path,
        sample_idx=args.sample_idx,
        target_class=args.target_class,
        antibiotic_names=args.antibiotics,
        output_dir=args.output_dir,
        layer_name=args.layer,
        interactive=not args.no_interactive
    )


if __name__ == '__main__':
    main()
