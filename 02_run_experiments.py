"""
Main experiment script
Compares Fuzzy-GraphSMOTE with baselines
"""

import sys
sys.path.append('./src')

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
import argparse
from torch_geometric.data import Data

from data_loader import load_cora, create_train_val_test_split
from fuzzy_graphsmote import FuzzyGraphSMOTE
from baselines import (VanillaGNN, GCN, SMOTE, FuzzySMOTE, GraphSMOTE,
                       ADASYN, BorderlineSMOTE, RandomUnderSampling,
                      ClassBalancedGNN, FocalLossGNN, ImGAGN)
from evaluation import FuzzyMetrics, print_comparison_table, geometric_mean_score


def run_single_experiment(dataset_name, method, data, fuzzy_labels, crisp_labels,
                          train_mask, val_mask, test_mask, config, device, verbose=True):
    """Run a single experiment with specified method."""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: {method} on {dataset_name}")
        print(f"{'='*80}")
    
    # ========================================================================
    # BASIC GNN BASELINES
    # ========================================================================
    
    if method == 'vanilla_gnn':
        model = VanillaGNN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                          config['num_layers'], config['dropout'])
        model.fit(data, fuzzy_labels, train_mask, val_mask, config['n_epochs'],
                 config['lr'], config['weight_decay'], config['patience'], device, verbose)
        fuzzy_pred, crisp_pred = model.predict(data, device)
    
    elif method == 'gcn':
        model = GCN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                   config['num_layers'], config['dropout'])
        model.fit(data, fuzzy_labels, train_mask, val_mask, config['n_epochs'],
                 config['lr'], config['weight_decay'], config['patience'], device, verbose)
        fuzzy_pred, crisp_pred = model.predict(data, device)
    
    # ========================================================================
    # LOSS-BASED METHODS
    # ========================================================================
    
    elif method == 'class_balanced':
        model = ClassBalancedGNN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                                config['num_layers'], config['dropout'])
        model.fit(data, fuzzy_labels, train_mask, val_mask, config['n_epochs'],
                 config['lr'], config['weight_decay'], config['patience'], device, verbose)
        fuzzy_pred, crisp_pred = model.predict(data, device)
    
    elif method == 'focal_loss':
        model = FocalLossGNN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                            config['num_layers'], config['dropout'])
        model.fit(data, fuzzy_labels, train_mask, val_mask, config['n_epochs'],
                 config['lr'], config['weight_decay'], config['patience'], device, verbose)
        fuzzy_pred, crisp_pred = model.predict(data, device)
    
    # ========================================================================
    # SAMPLING-BASED METHODS (NON-GRAPH)
    # ========================================================================
    
    elif method == 'rus':
        # Random Under-Sampling
        rus_model = RandomUnderSampling()
        X_train = data.x[train_mask].cpu().numpy()
        y_train = crisp_labels[train_mask].cpu().numpy()
        
        X_resampled, y_resampled = rus_model.fit_resample(X_train, y_train)
        
        # Create pseudo-graph (no edges)
        X_aug_torch = torch.FloatTensor(X_resampled)
        y_aug_torch = torch.LongTensor(y_resampled)
        
        aug_data = Data(x=X_aug_torch, edge_index=torch.empty((2, 0), dtype=torch.long))
        
        # Train GNN on undersampled data
        gnn_model = VanillaGNN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                              config['num_layers'], config['dropout'])
        
        y_aug_onehot = F.one_hot(y_aug_torch, num_classes=fuzzy_labels.shape[1]).float()
        
        gnn_model.fit(aug_data, y_aug_onehot,
                     torch.ones(len(X_resampled), dtype=torch.bool), None,
                     config['n_epochs'], config['lr'], config['weight_decay'],
                     config['patience'], device, verbose=False)
        
        fuzzy_pred, crisp_pred = gnn_model.predict(data, device)
    
    elif method == 'smote':
        # Standard SMOTE
        smote_model = SMOTE(k_neighbors=config['k_neighbors'], alpha=config['alpha'])
        X_train = data.x[train_mask].cpu().numpy()
        y_train = crisp_labels[train_mask].cpu().numpy()
        
        X_resampled, y_resampled = smote_model.fit_resample(X_train, y_train)
        
        X_aug_torch = torch.FloatTensor(X_resampled)
        y_aug_torch = torch.LongTensor(y_resampled)
        
        aug_data = Data(x=X_aug_torch, edge_index=torch.empty((2, 0), dtype=torch.long))
        
        gnn_model = VanillaGNN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                              config['num_layers'], config['dropout'])
        
        y_aug_onehot = F.one_hot(y_aug_torch, num_classes=fuzzy_labels.shape[1]).float()
        
        gnn_model.fit(aug_data, y_aug_onehot,
                     torch.ones(len(X_resampled), dtype=torch.bool), None,
                     config['n_epochs'], config['lr'], config['weight_decay'],
                     config['patience'], device, verbose=False)
        
        fuzzy_pred, crisp_pred = gnn_model.predict(data, device)
    
    elif method == 'adasyn':
        adasyn_model = ADASYN(k_neighbors=config['k_neighbors'], alpha=config['alpha'])
        X_train = data.x[train_mask].cpu().numpy()
        y_train = crisp_labels[train_mask].cpu().numpy()
        
        X_resampled, y_resampled = adasyn_model.fit_resample(X_train, y_train)
        
        X_aug_torch = torch.FloatTensor(X_resampled)
        y_aug_torch = torch.LongTensor(y_resampled)
        
        aug_data = Data(x=X_aug_torch, edge_index=torch.empty((2, 0), dtype=torch.long))
        
        gnn_model = VanillaGNN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                              config['num_layers'], config['dropout'])
        
        y_aug_onehot = F.one_hot(y_aug_torch, num_classes=fuzzy_labels.shape[1]).float()
        
        gnn_model.fit(aug_data, y_aug_onehot,
                     torch.ones(len(X_resampled), dtype=torch.bool), None,
                     config['n_epochs'], config['lr'], config['weight_decay'],
                     config['patience'], device, verbose=False)
        
        fuzzy_pred, crisp_pred = gnn_model.predict(data, device)
    
    elif method == 'borderline_smote':
        borderline_model = BorderlineSMOTE(k_neighbors=config['k_neighbors'], alpha=config['alpha'])
        X_train = data.x[train_mask].cpu().numpy()
        y_train = crisp_labels[train_mask].cpu().numpy()
        
        X_resampled, y_resampled = borderline_model.fit_resample(X_train, y_train)
        
        X_aug_torch = torch.FloatTensor(X_resampled)
        y_aug_torch = torch.LongTensor(y_resampled)
        
        aug_data = Data(x=X_aug_torch, edge_index=torch.empty((2, 0), dtype=torch.long))
        
        gnn_model = VanillaGNN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                              config['num_layers'], config['dropout'])
        
        y_aug_onehot = F.one_hot(y_aug_torch, num_classes=fuzzy_labels.shape[1]).float()
        
        gnn_model.fit(aug_data, y_aug_onehot,
                     torch.ones(len(X_resampled), dtype=torch.bool), None,
                     config['n_epochs'], config['lr'], config['weight_decay'],
                     config['patience'], device, verbose=False)
        
        fuzzy_pred, crisp_pred = gnn_model.predict(data, device)
    
    elif method == 'fuzzy_smote':
        fuzzy_smote_model = FuzzySMOTE(k_neighbors=config['k_neighbors'],
                                       alpha=config['alpha'], theta=config['theta'])
        X_train = data.x[train_mask].cpu().numpy()
        fuzzy_train = fuzzy_labels[train_mask].cpu().numpy()
        
        X_resampled, fuzzy_resampled = fuzzy_smote_model.fit_resample(X_train, fuzzy_train)
        
        X_aug_torch = torch.FloatTensor(X_resampled)
        fuzzy_aug_torch = torch.FloatTensor(fuzzy_resampled)
        
        aug_data = Data(x=X_aug_torch, edge_index=torch.empty((2, 0), dtype=torch.long))
        
        gnn_model = VanillaGNN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                              config['num_layers'], config['dropout'])
        
        gnn_model.fit(aug_data, fuzzy_aug_torch,
                     torch.ones(len(X_resampled), dtype=torch.bool), None,
                     config['n_epochs'], config['lr'], config['weight_decay'],
                     config['patience'], device, verbose=False)
        
        fuzzy_pred, crisp_pred = gnn_model.predict(data, device)
    
    # ========================================================================
    # GRAPH-BASED METHODS
    # ========================================================================
    
    elif method == 'graphsmote':
        model = GraphSMOTE(embedding_dim=config['hidden_dim'],
                          num_gnn_layers=config['num_layers'],
                          alpha=config['alpha'],
                          k_neighbors=config['k_neighbors'],
                          dropout=config['dropout'],
                          device=device)
        
        model.fit(data, crisp_labels, train_mask, val_mask,
                 config['n_epochs'], config['lr'], config['weight_decay'],
                 config['patience'], verbose)
        
        fuzzy_pred, crisp_pred = model.predict(data, device)
    
    elif method == 'imgagn':
        model = ImGAGN(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                      config['num_layers'], config['dropout'])
        
        model.fit(data, fuzzy_labels, train_mask, val_mask, config['n_epochs'],
                 config['lr'], config['weight_decay'], config['patience'], device, verbose)
        
        fuzzy_pred, crisp_pred = model.predict(data, device)
    
    elif method == 'graphens':
        model = GraphENS(data.x.shape[1], config['hidden_dim'], fuzzy_labels.shape[1],
                        config['num_layers'], config['dropout'], n_models=3)
        
        model.fit(data, fuzzy_labels, train_mask, val_mask, config['n_epochs'],
                 config['lr'], config['weight_decay'], config['patience'], device, verbose)
        
        fuzzy_pred, crisp_pred = model.predict(data, device)
    
    # ========================================================================
    # OUR METHOD
    # ========================================================================
    
    elif method == 'fuzzy_graphsmote':
        model = FuzzyGraphSMOTE(embedding_dim=config['hidden_dim'],
                               num_gnn_layers=config['num_layers'],
                               alpha=config['alpha'],
                               theta=config['theta'],
                               theta_neighbor=config['theta_neighbor'],
                               k_neighbors=config['k_neighbors'],
                               lambda_edge=config['lambda_edge'],
                               dropout=config['dropout'],
                               device=device)
        
        aug_data, aug_fuzzy_labels = model.fit(data, fuzzy_labels, train_mask, val_mask,
                                               config['n_pre_epochs'], config['n_joint_epochs'],
                                               config['lr'], config['weight_decay'],
                                               config['patience'], verbose)
        
        fuzzy_pred, crisp_pred = model.predict(data)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Evaluate on test set
    fuzzy_pred_test = fuzzy_pred[test_mask]
    fuzzy_true_test = fuzzy_labels[test_mask]
    crisp_true_test = crisp_labels[test_mask]
    
    metrics, cm = FuzzyMetrics.evaluate(fuzzy_pred_test, fuzzy_true_test,
                                       crisp_true_test, verbose=verbose)
    
    # Add G-Mean
    crisp_pred_test = crisp_pred[test_mask]
    gmean = geometric_mean_score(crisp_true_test.cpu().numpy(),
                                crisp_pred_test.cpu().numpy(),
                                num_classes=fuzzy_labels.shape[1])
    metrics['gmean'] = gmean
    
    if verbose:
        print(f"G-Mean: {gmean:.4f}")
    
    return metrics


def run_all_experiments(datasets=['cora'],
                       methods=['vanilla_gnn', 'class_balanced', 'graphsmote', 'fuzzy_graphsmote'],
                       n_runs=5, device='cuda' if torch.cuda.is_available() else 'cpu',
                       save_dir='./results'):
    """Run all experiments with multiple runs."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuration
    config = {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.5,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'patience': 20,
        'n_epochs': 200,
        'n_pre_epochs': 50,
        'n_joint_epochs': 150,
        'alpha': 1.5,
        'theta': 0.7,
        'theta_neighbor': 0.5,
        'k_neighbors': 8,
        'lambda_edge': 0.1
    }
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}\n")
        
        # Load dataset
        if dataset_name == 'cora':
            data, fuzzy_labels, crisp_labels = load_cora(alpha=0.4, seed=42)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Results for this dataset
        config_results = {method: [] for method in methods}
        
        for run in range(n_runs):
            print(f"\n--- Run {run+1}/{n_runs} ---")
            
            # Create train/val/test split
            train_mask, val_mask, test_mask = create_train_val_test_split(
                data.num_nodes, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42 + run
            )
            
            for method in methods:
                try:
                    metrics = run_single_experiment(
                        dataset_name, method, data, fuzzy_labels, crisp_labels,
                        train_mask, val_mask, test_mask, config, device,
                        verbose=(run == 0)  # Only verbose for first run
                    )
                    config_results[method].append(metrics)
                
                except Exception as e:
                    print(f"Error in {method}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Compute mean and std
        summary = {}
        for method, results_list in config_results.items():
            if len(results_list) == 0:
                continue
            
            summary[method] = {}
            metric_names = results_list[0].keys()
            
            for metric_name in metric_names:
                values = [r[metric_name] for r in results_list]
                summary[method][f"{metric_name}_mean"] = np.mean(values)
                summary[method][f"{metric_name}_std"] = np.std(values)
        
        all_results[dataset_name] = summary
        
        # Print comparison table
        print(f"\n{'='*80}")
        print(f"Summary for {dataset_name}")
        print(f"{'='*80}")
        
        comparison_dict = {}
        for method, stats in summary.items():
            comparison_dict[method] = {
                k.replace('_mean', ''): v 
                for k, v in stats.items() 
                if k.endswith('_mean')
            }
        
        print_comparison_table(comparison_dict)
        
        # Print with std
        print("\nDetailed Results (Mean ± Std):")
        for method, stats in summary.items():
            print(f"\n{method}:")
            key_metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'gmean']
            for metric in key_metrics:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                if mean_key in stats:
                    print(f"  {metric}: {stats[mean_key]:.4f} ± {stats[std_key]:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(save_dir, f"results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run Fuzzy-GraphSMOTE experiments')
    parser.add_argument('--datasets', nargs='+', default=['cora'], 
                        choices=['cora'])
    parser.add_argument('--methods', nargs='+', 
                        default=['vanilla_gnn', 'gcn',                                    # GNN
                                 'class_balanced', 'focal_loss',                          # Loss
                                 'rus', 'smote', 'adasyn', 'borderline_smote', 'fuzzy_smote',  # Sampling
                                 'graphsmote', 'imgagn',                      # Graph
                                 'fuzzy_graphsmote'                                       # OURS
                    ],
                        help='Methods to compare')
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FUZZY-GRAPHSMOTE EXPERIMENTS - COMPLETE VERSION")
    print("="*80)
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    print(f"Runs per config: {args.n_runs}")
    print(f"Device: {args.device}")
    print("="*80)
    
    results = run_all_experiments(args.datasets, args.methods, args.n_runs, args.device, args.save_dir)
    
    print("\n✓ All experiments completed!")


if __name__ == "__main__":
    main()
