"""
Evaluation metrics for fuzzy and crisp classification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, balanced_accuracy_score
)
from typing import Dict, Tuple, Optional


class FuzzyMetrics:
    """
    Evaluation metrics for fuzzy node classification.
    """
    
    @staticmethod
    def fuzzy_accuracy(
        fuzzy_pred: torch.Tensor,
        fuzzy_true: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        Fuzzy accuracy based on membership agreement.
        
        Args:
            fuzzy_pred: Predicted fuzzy memberships [num_nodes, num_classes]
            fuzzy_true: True fuzzy memberships [num_nodes, num_classes]
            threshold: Membership threshold for considering a match
            
        Returns:
            fuzzy_accuracy: Proportion of nodes where predictions match true labels
        """
        # Compute cosine similarity between predicted and true memberships
        similarities = F.cosine_similarity(
            fuzzy_pred, fuzzy_true, dim=1
        )
        
        # Consider a match if similarity > threshold
        matches = (similarities > threshold).float()
        
        return matches.mean().item()
    
    @staticmethod
    def crisp_metrics(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        num_classes: int = 2
    ) -> Dict[str, float]:
        """
        Standard classification metrics on crisp labels.
        
        Args:
            y_pred: Predicted crisp labels [num_nodes]
            y_true: True crisp labels [num_nodes]
            num_classes: Number of classes
            
        Returns:
            metrics: Dictionary of metric names and values
        """
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        
        # Overall metrics
        acc = accuracy_score(y_true_np, y_pred_np)
        balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_np, y_pred_np, average=None, zero_division=0
        )
        
        # Macro-averaged metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='macro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_np, y_pred_np)
        
        metrics = {
            'accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
        }
        
        # Add per-class metrics
        for c in range(num_classes):
            metrics[f'precision_class_{c}'] = precision[c] if c < len(precision) else 0.0
            metrics[f'recall_class_{c}'] = recall[c] if c < len(recall) else 0.0
            metrics[f'f1_class_{c}'] = f1[c] if c < len(f1) else 0.0
        
        # Minority class (class 0 for Cora, class 1 for MUSAE)
        # We'll report both and let the user decide
        metrics['minority_f1_c0'] = f1[0] if 0 < len(f1) else 0.0
        metrics['minority_f1_c1'] = f1[1] if 1 < len(f1) else 0.0
        
        return metrics, cm
    
    @staticmethod
    def fuzzy_cross_entropy(
        fuzzy_pred: torch.Tensor,
        fuzzy_true: torch.Tensor
    ) -> float:
        """
        Fuzzy cross-entropy loss.
        
        Args:
            fuzzy_pred: Predicted fuzzy memberships [num_nodes, num_classes]
            fuzzy_true: True fuzzy memberships [num_nodes, num_classes]
            
        Returns:
            loss: Fuzzy cross-entropy
        """
        loss = -(fuzzy_true * torch.log(fuzzy_pred + 1e-8)).sum(dim=1).mean()
        return loss.item()
    
    @staticmethod
    def evaluate(
        fuzzy_pred: torch.Tensor,
        fuzzy_true: torch.Tensor,
        crisp_true: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation with both fuzzy and crisp metrics.
        
        Args:
            fuzzy_pred: Predicted fuzzy memberships [num_nodes, num_classes]
            fuzzy_true: True fuzzy memberships [num_nodes, num_classes]
            crisp_true: True crisp labels [num_nodes]
            verbose: Print results
            
        Returns:
            all_metrics: Dictionary with all metrics
        """
        import torch.nn.functional as F
        
        # Convert to crisp predictions
        crisp_pred = fuzzy_pred.argmax(dim=1)
        
        # Crisp metrics
        crisp_metrics_dict, cm = FuzzyMetrics.crisp_metrics(
            crisp_pred, crisp_true, num_classes=fuzzy_pred.shape[1]
        )
        
        # Fuzzy metrics
        fuzzy_acc = FuzzyMetrics.fuzzy_accuracy(fuzzy_pred, fuzzy_true)
        fuzzy_ce = FuzzyMetrics.fuzzy_cross_entropy(fuzzy_pred, fuzzy_true)
        
        all_metrics = {
            'fuzzy_accuracy': fuzzy_acc,
            'fuzzy_cross_entropy': fuzzy_ce,
            **crisp_metrics_dict
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            print(f"Fuzzy Accuracy: {fuzzy_acc:.4f}")
            print(f"Fuzzy Cross-Entropy: {fuzzy_ce:.4f}")
            print(f"\nCrisp Accuracy: {all_metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy: {all_metrics['balanced_accuracy']:.4f}")
            print(f"Macro F1-Score: {all_metrics['macro_f1']:.4f}")
            print(f"Macro Precision: {all_metrics['macro_precision']:.4f}")
            print(f"Macro Recall: {all_metrics['macro_recall']:.4f}")
            
            print(f"\nPer-class metrics:")
            for c in range(fuzzy_pred.shape[1]):
                print(f"  Class {c}:")
                print(f"    Precision: {all_metrics[f'precision_class_{c}']:.4f}")
                print(f"    Recall: {all_metrics[f'recall_class_{c}']:.4f}")
                print(f"    F1-Score: {all_metrics[f'f1_class_{c}']:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(cm)
            print("=" * 60)
        
        return all_metrics, cm


def print_comparison_table(results_dict: Dict[str, Dict[str, float]]):
    """
    Print a comparison table of multiple methods.
    
    Args:
        results_dict: Dictionary mapping method names to metric dictionaries
    """
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    
    # Key metrics to display
    key_metrics = [
        'accuracy',
        'balanced_accuracy',
        'macro_f1',
        'macro_precision',
        'macro_recall',
        'minority_f1_c0',
        'minority_f1_c1'
    ]
    
    # Header
    header = f"{'Method':<25}"
    for metric in key_metrics:
        header += f"{metric:<20}"
    print(header)
    print("-" * 100)
    
    # Rows
    for method_name, metrics in results_dict.items():
        row = f"{method_name:<25}"
        for metric in key_metrics:
            value = metrics.get(metric, 0.0)
            row += f"{value:<20.4f}"
        print(row)
    
    print("=" * 100)


# G-Mean metric for imbalanced classification
def geometric_mean_score(y_true, y_pred, num_classes=2):
    """
    Compute G-Mean (geometric mean of per-class recalls).
    Important metric for imbalanced classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        gmean: Geometric mean of recalls
    """
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Geometric mean
    gmean = np.prod(recall) ** (1.0 / len(recall))
    
    return gmean


# Example usage
if __name__ == "__main__":
    # Simulate some predictions
    num_nodes = 100
    num_classes = 2
    
    fuzzy_pred = torch.softmax(torch.randn(num_nodes, num_classes), dim=1)
    fuzzy_true = torch.softmax(torch.randn(num_nodes, num_classes), dim=1)
    crisp_true = fuzzy_true.argmax(dim=1)
    
    print("Testing evaluation metrics...")
    
    metrics, cm = FuzzyMetrics.evaluate(
        fuzzy_pred, fuzzy_true, crisp_true, verbose=True
    )
    
    # Test comparison table
    print("\nTesting comparison table...")
    
    results = {
        'Method A': metrics,
        'Method B': {k: v * 0.9 for k, v in metrics.items()},
        'Method C': {k: v * 1.1 for k, v in metrics.items()},
    }
    
    print_comparison_table(results)
