"""
Visualization utilities for Fuzzy-GraphSMOTE results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
from torch_geometric.utils import to_networkx


def plot_fuzzy_membership_distribution(
    fuzzy_labels: torch.Tensor,
    class_names: list = None,
    save_path: str = None
):
    """
    Plot distribution of fuzzy memberships for each class.
    """
    num_classes = fuzzy_labels.shape[1]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    fig, axes = plt.subplots(1, num_classes, figsize=(7*num_classes, 5))
    if num_classes == 1:
        axes = [axes]
    
    for c in range(num_classes):
        memberships = fuzzy_labels[:, c].numpy()
        
        axes[c].hist(memberships, bins=50, alpha=0.7, color=f'C{c}', edgecolor='black')
        axes[c].axvline(x=0.6, color='red', linestyle='--', linewidth=2, 
                       label='Min membership (0.6)')
        axes[c].set_xlabel(f'Membership to {class_names[c]}', fontsize=12)
        axes[c].set_ylabel('Frequency', fontsize=12)
        axes[c].set_title(f'Distribution: {class_names[c]}', fontsize=14)
        axes[c].legend()
        axes[c].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = memberships.mean()
        std_val = memberships.std()
        axes[c].text(0.05, 0.95, f'μ = {mean_val:.3f}\nσ = {std_val:.3f}',
                    transform=axes[c].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_fuzzy_membership_space(
    fuzzy_labels: torch.Tensor,
    crisp_labels: torch.Tensor,
    framework: str = 'probabilistic',
    save_path: str = None
):
    """
    2D visualization of fuzzy membership space (binary case).
    """
    if fuzzy_labels.shape[1] != 2:
        print("Warning: This visualization is designed for binary classification")
        return
    
    plt.figure(figsize=(10, 8))
    
    mu_c0 = fuzzy_labels[:, 0].numpy()
    mu_c1 = fuzzy_labels[:, 1].numpy()
    
    scatter = plt.scatter(
        mu_c0, mu_c1,
        c=crisp_labels.numpy(),
        cmap='coolwarm',
        alpha=0.6,
        s=20,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add constraint lines
    plt.axvline(x=0.6, color='blue', linestyle='--', linewidth=2, alpha=0.7, 
               label='μ(C₀) = 0.6')
    plt.axhline(y=0.6, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label='μ(C₁) = 0.6')
    
    if framework == 'probabilistic':
        plt.plot([0, 1], [1, 0], 'gray', linestyle=':', linewidth=2, alpha=0.5,
                label='μ(C₀) + μ(C₁) = 1')
    
    plt.xlabel('Membership to Class 0', fontsize=12)
    plt.ylabel('Membership to Class 1', fontsize=12)
    plt.title(f'Fuzzy Membership Space ({framework.title()} Framework)', fontsize=14)
    plt.colorbar(scatter, label='Crisp Label')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_node_embeddings_tsne(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    node_types: list = None,
    title: str = 'Node Embeddings (t-SNE)',
    save_path: str = None
):
    """
    Visualize node embeddings using t-SNE.
    
    Args:
        embeddings: Node embeddings [num_nodes, dim]
        labels: Node labels [num_nodes]
        node_types: Optional list indicating 'original' or 'synthetic'
        title: Plot title
        save_path: Path to save figure
    """
    print("Computing t-SNE projection...")
    
    embeddings_np = embeddings.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    if node_types is not None:
        # Plot with node types
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: By node type
        for nt, color in [('original', 'blue'), ('synthetic', 'red')]:
            mask = [t == nt for t in node_types]
            axes[0].scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color,
                label=nt.title(),
                alpha=0.6,
                s=30
            )
        
        axes[0].set_title('Original vs Synthetic Nodes', fontsize=14)
        axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
        axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Right: By class
        scatter = axes[1].scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels_np,
            cmap='viridis',
            alpha=0.6,
            s=30,
            edgecolors='black',
            linewidth=0.3
        )
        
        axes[1].set_title('Node Classes', fontsize=14)
        axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
        axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
        plt.colorbar(scatter, ax=axes[1], label='Class')
        axes[1].grid(True, alpha=0.3)
    else:
        # Simple plot by class
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels_np,
            cmap='viridis',
            alpha=0.6,
            s=30,
            edgecolors='black',
            linewidth=0.3
        )
        
        plt.title(title, fontsize=14)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.colorbar(scatter, label='Class')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    save_path: str = None
):
    """
    Plot confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_method_comparison(
    results_dict: dict,
    metrics: list = ['accuracy', 'balanced_accuracy', 'macro_f1', 'gmean'],
    save_path: str = None
):
    """
    Bar plot comparing different methods across multiple metrics.
    
    Args:
        results_dict: Dict mapping method names to metric dictionaries
        metrics: List of metrics to plot
        save_path: Path to save figure
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    methods = list(results_dict.keys())
    x = np.arange(len(methods))
    width = 0.6
    
    for idx, metric in enumerate(metrics):
        values = [results_dict[m].get(metric, 0) for m in methods]
        stds = [results_dict[m].get(f'{metric}_std', 0) for m in methods]
        
        bars = axes[idx].bar(x, values, width, yerr=stds, capsize=5, alpha=0.7)
        
        # Color the best method
        best_idx = np.argmax(values)
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(1.0)
        
        axes[idx].set_ylabel('Score', fontsize=12)
        axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=14)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(methods, rotation=45, ha='right')
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution(
    labels: torch.Tensor,
    class_names: list = None,
    title: str = 'Class Distribution',
    save_path: str = None
):
    """
    Plot class distribution as a bar chart.
    """
    unique_labels, counts = torch.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels.numpy()]
    
    plt.figure(figsize=(8, 6))
    
    bars = plt.bar(range(len(unique_labels)), counts.numpy(), alpha=0.7)
    
    # Color minority class differently
    minority_idx = counts.argmin()
    bars[minority_idx].set_color('red')
    bars[minority_idx].set_alpha(1.0)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Nodes', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(range(len(unique_labels)), class_names)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count.item()}',
                ha='center', va='bottom', fontsize=11)
    
    # Add imbalance ratio
    if len(counts) == 2:
        ratio = min(counts).item() / max(counts).item()
        plt.text(0.95, 0.95, f'Imbalance Ratio: {ratio:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(
    train_losses: list,
    val_losses: list = None,
    title: str = 'Training Curves',
    save_path: str = None
):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Example usage
if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_fuzzy_membership_distribution")
    print("  - plot_fuzzy_membership_space")
    print("  - plot_node_embeddings_tsne")
    print("  - plot_confusion_matrix")
    print("  - plot_method_comparison")
    print("  - plot_class_distribution")
    print("  - plot_training_curves")
