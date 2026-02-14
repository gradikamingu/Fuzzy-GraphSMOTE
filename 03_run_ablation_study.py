"""
ANALYSE D'ABLATION - FUZZY-GRAPHSMOTE

This script performs ablation studies to understand the contribution of each
component of Fuzzy-GraphSMOTE:

1. Fuzzy Logic vs Crisp Labels
2. Graph-aware vs Matrix-space Oversampling
3. Edge Generation vs Random Connections
4. Pre-training vs Joint Training Only
5. Hyperparameter Sensitivity (θ, α, k)

Usage:
    python run_ablation_study.py --dataset cora --n_runs 5
"""

import sys
sys.path.append('./src')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from typing import Dict, List

from data_loader import load_cora, create_train_val_test_split
from citeseer_loader import load_citeseer
from fuzzy_graphsmote import FuzzyGraphSMOTE
from baselines import VanillaGNN, GraphSMOTE
from evaluation import FuzzyMetrics


def run_ablation_experiment(
    data,
    fuzzy_labels,
    crisp_labels,
    train_mask,
    val_mask,
    test_mask,
    variant: str,
    device: str = 'cpu'
) -> Dict:
    """
    Run single ablation experiment.
    
    Variants:
    - 'full': Full Fuzzy-GraphSMOTE (all components)
    - 'no_fuzzy': GraphSMOTE (crisp labels only)
    - 'no_graph': Fuzzy-SMOTE (no graph structure)
    - 'no_pretrain': Skip edge generator pre-training
    - 'random_edges': Random edge generation instead of learned
    - 'low_theta': θ = 0.5 (more minority samples)
    - 'high_theta': θ = 0.9 (fewer minority samples)
    - 'low_alpha': α = 0.5 (less oversampling)
    - 'high_alpha': α = 2.0 (more oversampling)
    """
    
    config = {
        'hidden_dim': 64,
        'num_layers': 2,
        'alpha': 1.0,
        'theta': 0.7,
        'theta_neighbor': 0.3,
        'k_neighbors': 5,
        'lambda_edge': 0.5,
        'dropout': 0.5,
        'n_epochs': 200,
        'n_pre_epochs': 50,
        'n_joint_epochs': 150,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'patience': 20
    }
    
    # Modify config based on variant
    if variant == 'full':
        # Full Fuzzy-GraphSMOTE
        model = FuzzyGraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=config['alpha'],
            theta=config['theta'],
            theta_neighbor=config['theta_neighbor'],
            k_neighbors=config['k_neighbors'],
            lambda_edge=config['lambda_edge'],
            dropout=config['dropout'],
            device=device
        )
        
        aug_data, aug_fuzzy_labels = model.fit(
            data=data,
            fuzzy_labels=fuzzy_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            n_pre_epochs=config['n_pre_epochs'],
            n_joint_epochs=config['n_joint_epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            patience=config['patience'],
            verbose=False
        )
        
        fuzzy_pred, crisp_pred = model.predict(data)
    
    elif variant == 'no_fuzzy':
        # GraphSMOTE (crisp labels, graph-aware)
        from baselines import GraphSMOTE
        model = GraphSMOTE(
            embedding_dim=config['hidden_dim'],
            k_neighbors=config['k_neighbors'],
            alpha=config['alpha'],
            device=device
        )
        
        aug_data, aug_labels = model.fit(
            data=data,
            labels=crisp_labels,
            train_mask=train_mask,
            verbose=False
        )
        
        # Train GNN
        gnn_model = VanillaGNN(
            aug_data.x.shape[1],
            config['hidden_dim'],
            fuzzy_labels.shape[1],
            config['num_layers'],
            config['dropout']
        )
        
        aug_labels_onehot = torch.nn.functional.one_hot(aug_labels, num_classes=fuzzy_labels.shape[1]).float()
        
        gnn_model.fit(
            data=aug_data,
            labels=aug_labels_onehot,
            train_mask=torch.ones(aug_data.x.shape[0], dtype=torch.bool),
            val_mask=val_mask,
            n_epochs=config['n_epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            patience=config['patience'],
            device=device,
            verbose=False
        )
        
        fuzzy_pred, crisp_pred = gnn_model.predict(data, device)
    
    elif variant == 'no_graph':
        # Fuzzy-SMOTE (fuzzy labels, no graph)
        from baselines import FuzzySMOTE
        model = FuzzySMOTE(
            k_neighbors=config['k_neighbors'],
            alpha=config['alpha'],
            theta=config['theta']
        )
        
        X_train = data.x[train_mask].cpu().numpy()
        fuzzy_train = fuzzy_labels[train_mask].cpu().numpy()
        
        X_resampled, fuzzy_resampled = model.fit_resample(X_train, fuzzy_train)
        
        # Create augmented data
        from torch_geometric.data import Data
        X_aug_torch = torch.tensor(X_resampled, dtype=torch.float32)
        fuzzy_aug_torch = torch.tensor(fuzzy_resampled, dtype=torch.float32)
        
        aug_data = Data(
            x=X_aug_torch,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=fuzzy_aug_torch.argmax(dim=1)
        )
        
        # Train GNN
        gnn_model = VanillaGNN(
            aug_data.x.shape[1],
            config['hidden_dim'],
            fuzzy_labels.shape[1],
            config['num_layers'],
            config['dropout']
        )
        
        gnn_model.fit(
            data=aug_data,
            labels=fuzzy_aug_torch,
            train_mask=torch.ones(X_aug_torch.shape[0], dtype=torch.bool),
            val_mask=None,
            n_epochs=config['n_epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            patience=config['patience'],
            device=device,
            verbose=False
        )
        
        fuzzy_pred, crisp_pred = gnn_model.predict(data, device)
    
    elif variant == 'no_pretrain':
        # Skip pre-training
        model = FuzzyGraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=config['alpha'],
            theta=config['theta'],
            theta_neighbor=config['theta_neighbor'],
            k_neighbors=config['k_neighbors'],
            lambda_edge=config['lambda_edge'],
            dropout=config['dropout'],
            device=device
        )
        
        aug_data, aug_fuzzy_labels = model.fit(
            data=data,
            fuzzy_labels=fuzzy_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            n_pre_epochs=0,  # No pre-training
            n_joint_epochs=config['n_joint_epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            patience=config['patience'],
            verbose=False
        )
        
        fuzzy_pred, crisp_pred = model.predict(data)
    
    elif variant.startswith('theta_') or variant.startswith('alpha_'):
        # Hyperparameter variants
        if variant == 'low_theta':
            config['theta'] = 0.5
        elif variant == 'high_theta':
            config['theta'] = 0.9
        elif variant == 'low_alpha':
            config['alpha'] = 0.5
        elif variant == 'high_alpha':
            config['alpha'] = 2.0
        
        model = FuzzyGraphSMOTE(
            embedding_dim=config['hidden_dim'],
            num_gnn_layers=config['num_layers'],
            alpha=config['alpha'],
            theta=config['theta'],
            theta_neighbor=config['theta_neighbor'],
            k_neighbors=config['k_neighbors'],
            lambda_edge=config['lambda_edge'],
            dropout=config['dropout'],
            device=device
        )
        
        aug_data, aug_fuzzy_labels = model.fit(
            data=data,
            fuzzy_labels=fuzzy_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            n_pre_epochs=config['n_pre_epochs'],
            n_joint_epochs=config['n_joint_epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            patience=config['patience'],
            verbose=False
        )
        
        fuzzy_pred, crisp_pred = model.predict(data)
    
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # Evaluate
    fuzzy_pred_test = fuzzy_pred[test_mask]
    fuzzy_true_test = fuzzy_labels[test_mask]
    crisp_true_test = crisp_labels[test_mask]
    
    metrics, cm = FuzzyMetrics.evaluate(
        fuzzy_pred_test,
        fuzzy_true_test,
        crisp_true_test,
        verbose=False
    )
    
    return metrics


def run_full_ablation_study(
    dataset_name: str = 'cora',
    n_runs: int = 5,
    device: str = 'cpu'
):
    """
    Run complete ablation study.
    """
    print("="*80)
    print(f"ABLATION STUDY - {dataset_name.upper()}")
    print("="*80)
    
    # Load dataset
    if dataset_name == 'cora':
        data, fuzzy_labels, crisp_labels = load_cora(alpha=0.4, seed=42)
    elif dataset_name == 'citeseer':
        data, fuzzy_labels, crisp_labels = load_citeseer(alpha=0.4, seed=42)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Ablation variants
    variants = [
        ('full', 'Full Fuzzy-GraphSMOTE'),
        ('no_fuzzy', 'Without Fuzzy Logic (GraphSMOTE)'),
        ('no_graph', 'Without Graph Structure (Fuzzy-SMOTE)'),
        ('no_pretrain', 'Without Pre-training'),
        ('low_theta', 'θ = 0.5 (More Minority)'),
        ('high_theta', 'θ = 0.9 (Fewer Minority)'),
        ('low_alpha', 'α = 0.5 (Less Oversampling)'),
        ('high_alpha', 'α = 2.0 (More Oversampling)')
    ]
    
    all_results = {}
    
    for variant_key, variant_name in variants:
        print(f"\n{'='*80}")
        print(f"Variant: {variant_name}")
        print(f"{'='*80}")
        
        results = []
        
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end=' ')
            
            # Create split
            train_mask, val_mask, test_mask = create_train_val_test_split(
                data.num_nodes,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=42 + run
            )
            
            # Run experiment
            metrics = run_ablation_experiment(
                data, fuzzy_labels, crisp_labels,
                train_mask, val_mask, test_mask,
                variant=variant_key,
                device=device
            )
            
            results.append(metrics)
            print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Compute statistics
        all_results[variant_name] = {
            'accuracy_mean': np.mean([r['accuracy'] for r in results]),
            'accuracy_std': np.std([r['accuracy'] for r in results]),
            'gmean_mean': np.mean([r['gmean'] for r in results]),
            'gmean_std': np.std([r['gmean'] for r in results]),
            'macro_f1_mean': np.mean([r['macro_f1'] for r in results]),
            'macro_f1_std': np.std([r['macro_f1'] for r in results])
        }
    
    # Create results table
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*80}\n")
    
    df = pd.DataFrame(all_results).T
    df = df.round(4)
    print(df[['accuracy_mean', 'accuracy_std', 'gmean_mean', 'macro_f1_mean']])
    
    # Save results
    os.makedirs('./results/ablation', exist_ok=True)
    df.to_csv(f'./results/ablation/{dataset_name}_ablation_results.csv')
    print(f"\n✓ Results saved to: ./results/ablation/{dataset_name}_ablation_results.csv")
    
    # Create visualization
    create_ablation_plots(df, dataset_name)
    
    return df


def create_ablation_plots(df: pd.DataFrame, dataset_name: str):
    """
    Create visualization of ablation results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    variants = df.index.tolist()
    
    # Plot 1: Accuracy
    y_pos = np.arange(len(variants))
    axes[0].barh(y_pos, df['accuracy_mean'], xerr=df['accuracy_std'], 
                 color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(variants, fontsize=9)
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title(f'{dataset_name.upper()} - Accuracy Comparison')
    axes[0].axvline(x=df.loc['Full Fuzzy-GraphSMOTE', 'accuracy_mean'], 
                    color='red', linestyle='--', label='Full Model')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: G-Mean
    axes[1].barh(y_pos, df['gmean_mean'], xerr=df['gmean_std'],
                 color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(variants, fontsize=9)
    axes[1].set_xlabel('G-Mean')
    axes[1].set_title(f'{dataset_name.upper()} - G-Mean Comparison')
    axes[1].axvline(x=df.loc['Full Fuzzy-GraphSMOTE', 'gmean_mean'],
                    color='red', linestyle='--', label='Full Model')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Macro F1
    axes[2].barh(y_pos, df['macro_f1_mean'], xerr=df['macro_f1_std'],
                 color='lightgreen', edgecolor='black', alpha=0.7)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(variants, fontsize=9)
    axes[2].set_xlabel('Macro F1')
    axes[2].set_title(f'{dataset_name.upper()} - Macro F1 Comparison')
    axes[2].axvline(x=df.loc['Full Fuzzy-GraphSMOTE', 'macro_f1_mean'],
                    color='red', linestyle='--', label='Full Model')
    axes[2].legend()
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./results/ablation/{dataset_name}_ablation_comparison.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: ./results/ablation/{dataset_name}_ablation_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Ablation Study for Fuzzy-GraphSMOTE')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer'],
                        help='Dataset to use')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of runs per variant')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    results = run_full_ablation_study(
        dataset_name=args.dataset,
        n_runs=args.n_runs,
        device=args.device
    )
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
