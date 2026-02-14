"""
Baseline methods for imbalanced graph classification.

Complete set including:
- Vanilla GNN, GCN
- SMOTE, Fuzzy-SMOTE, ADASYN, Borderline-SMOTE
- GraphSMOTE, ImGAGN
- Class-balanced Loss, Focal Loss
- Random Under-Sampling (RUS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.neighbors import NearestNeighbors
from collections import Counter


# ============================================================================
# BASIC GNN BASELINES
# ============================================================================

class VanillaGNN(nn.Module):
    """Vanilla GNN without any imbalance handling."""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
    
    def fit(self, data, labels, train_mask, val_mask=None, n_epochs=200,
            lr=0.01, weight_decay=5e-4, patience=20, device='cpu', verbose=True):
        self.to(device)
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            
            if labels.dim() == 2:  # Fuzzy labels
                loss = -(labels[train_mask] * torch.log(F.softmax(out[train_mask], dim=1) + 1e-8)).sum(dim=1).mean()
            else:  # Crisp labels
                loss = F.cross_entropy(out[train_mask], labels[train_mask])
            
            loss.backward()
            optimizer.step()
            
            if val_mask is not None:
                self.eval()
                with torch.no_grad():
                    val_out = self(data.x, data.edge_index)
                    if labels.dim() == 2:
                        val_loss = -(labels[val_mask] * torch.log(F.softmax(val_out[val_mask], dim=1) + 1e-8)).sum(dim=1).mean()
                    else:
                        val_loss = F.cross_entropy(val_out[val_mask], labels[val_mask])
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def predict(self, data, device='cpu'):
        self.eval()
        data = data.to(device)
        with torch.no_grad():
            out = self(data.x, data.edge_index)
            fuzzy_pred = F.softmax(out, dim=1)
            crisp_pred = fuzzy_pred.argmax(dim=1)
        return fuzzy_pred, crisp_pred


class GCN(VanillaGNN):
    """GCN baseline."""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__(in_channels, hidden_channels, num_classes, num_layers, dropout)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))


# ============================================================================
# LOSS-BASED METHODS
# ============================================================================

class ClassBalancedGNN(VanillaGNN):
    """GNN with class-balanced loss (inverse frequency weighting)."""
    
    def fit(self, data, labels, train_mask, val_mask=None, n_epochs=200,
            lr=0.01, weight_decay=5e-4, patience=20, device='cpu', verbose=True):
        self.to(device)
        data = data.to(device)
        labels = labels.to(device)
        
        # Calculate class weights (inverse frequency)
        if labels.dim() == 2:
            crisp_labels = labels.argmax(dim=1)
        else:
            crisp_labels = labels
        
        class_counts = torch.bincount(crisp_labels[train_mask], minlength=labels.shape[1] if labels.dim() == 2 else int(labels.max() + 1))
        class_weights = 1.0 / (class_counts.float() + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)
        
        if verbose:
            print(f"  Class weights: {class_weights.cpu().numpy()}")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            
            if labels.dim() == 2:
                # Fuzzy cross-entropy with class weights
                probs = F.softmax(out[train_mask], dim=1)
                loss = -(labels[train_mask] * torch.log(probs + 1e-8) * class_weights.unsqueeze(0)).sum(dim=1).mean()
            else:
                loss = F.cross_entropy(out[train_mask], labels[train_mask], weight=class_weights)
            
            loss.backward()
            optimizer.step()
            
            if val_mask is not None:
                self.eval()
                with torch.no_grad():
                    val_out = self(data.x, data.edge_index)
                    if labels.dim() == 2:
                        val_probs = F.softmax(val_out[val_mask], dim=1)
                        val_loss = -(labels[val_mask] * torch.log(val_probs + 1e-8)).sum(dim=1).mean()
                    else:
                        val_loss = F.cross_entropy(val_out[val_mask], labels[val_mask])
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        """
        pred: [N, C] logits
        target: [N] or [N, C] (crisp or fuzzy)
        """
        probs = F.softmax(pred, dim=1)
        
        if target.dim() == 1:
            # Crisp labels
            ce_loss = F.cross_entropy(pred, target, reduction='none')
            pt = probs[range(len(target)), target]
        else:
            # Fuzzy labels
            ce_loss = -(target * torch.log(probs + 1e-8)).sum(dim=1)
            pt = (probs * target).sum(dim=1)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class FocalLossGNN(VanillaGNN):
    """GNN with Focal Loss."""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__(in_channels, hidden_channels, num_classes, num_layers, dropout)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes)
    
    def fit(self, data, labels, train_mask, val_mask=None, n_epochs=200,
            lr=0.01, weight_decay=5e-4, patience=20, device='cpu', verbose=True):
        self.to(device)
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            
            loss = self.focal_loss(out[train_mask], labels[train_mask])
            
            loss.backward()
            optimizer.step()
            
            if val_mask is not None:
                self.eval()
                with torch.no_grad():
                    val_out = self(data.x, data.edge_index)
                    val_loss = self.focal_loss(val_out[val_mask], labels[val_mask])
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break


# ============================================================================
# SAMPLING-BASED METHODS
# ============================================================================

class RandomUnderSampling:
    """Random Under-Sampling of majority class."""
    
    def fit_resample(self, X, y):
        """
        X: numpy array [N, F]
        y: numpy array [N]
        """
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[counts.argmin()]
        minority_size = counts.min()
        
        minority_idx = np.where(y == minority_class)[0]
        majority_idx = np.where(y != minority_class)[0]
        
        # Sample from majority
        sampled_majority = np.random.choice(majority_idx, minority_size, replace=False)
        
        balanced_idx = np.concatenate([minority_idx, sampled_majority])
        np.random.shuffle(balanced_idx)
        
        return X[balanced_idx], y[balanced_idx]


class SMOTE:
    """Standard SMOTE (Synthetic Minority Over-sampling Technique)."""
    
    def __init__(self, k_neighbors=5, alpha=1.0):
        self.k_neighbors = k_neighbors
        self.alpha = alpha
    
    def fit_resample(self, X, y):
        """
        X: numpy array [N, F]
        y: numpy array [N]
        """
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[counts.argmin()]
        minority_size = counts.min()
        majority_size = counts.max()
        
        minority_idx = np.where(y == minority_class)[0]
        X_minority = X[minority_idx]
        
        # Number of synthetic samples
        n_synthetic = int(self.alpha * (majority_size - minority_size))
        
        if n_synthetic == 0 or len(X_minority) < 2:
            return X, y
        
        # k-NN
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_minority)))
        nbrs.fit(X_minority)
        
        synthetic_samples = []
        for _ in range(n_synthetic):
            idx = np.random.randint(0, len(X_minority))
            distances, indices = nbrs.kneighbors(X_minority[idx:idx+1])
            
            # Remove self
            neighbor_indices = indices[0][indices[0] != idx]
            if len(neighbor_indices) == 0:
                continue
            
            neighbor_idx = np.random.choice(neighbor_indices)
            
            # Linear interpolation
            delta = np.random.uniform(0, 1)
            synthetic = X_minority[idx] + delta * (X_minority[neighbor_idx] - X_minority[idx])
            synthetic_samples.append(synthetic)
        
        if len(synthetic_samples) == 0:
            return X, y
        
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(len(X_synthetic), minority_class)
        
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.concatenate([y, y_synthetic])
        
        return X_resampled, y_resampled


class ADASYN:
    """ADASYN (Adaptive Synthetic Sampling)."""
    
    def __init__(self, k_neighbors=5, alpha=1.0):
        self.k_neighbors = k_neighbors
        self.alpha = alpha
    
    def fit_resample(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[counts.argmin()]
        minority_size = counts.min()
        majority_size = counts.max()
        
        minority_idx = np.where(y == minority_class)[0]
        X_minority = X[minority_idx]
        
        n_synthetic_total = int(self.alpha * (majority_size - minority_size))
        
        if n_synthetic_total == 0 or len(X_minority) < 2:
            return X, y
        
        # Calculate difficulty (ratio of majority neighbors)
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X)))
        nbrs.fit(X)
        
        difficulties = []
        for idx in minority_idx:
            distances, indices = nbrs.kneighbors(X[idx:idx+1])
            neighbor_labels = y[indices[0]]
            ratio = (neighbor_labels != minority_class).sum() / len(neighbor_labels)
            difficulties.append(ratio)
        
        difficulties = np.array(difficulties)
        difficulties = difficulties / (difficulties.sum() + 1e-8)
        
        # Generate synthetic samples proportional to difficulty
        nbrs_minority = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_minority)))
        nbrs_minority.fit(X_minority)
        
        synthetic_samples = []
        for local_idx, global_idx in enumerate(minority_idx):
            n_samples = int(difficulties[local_idx] * n_synthetic_total)
            
            for _ in range(n_samples):
                distances, indices = nbrs_minority.kneighbors(X[global_idx:global_idx+1])
                neighbor_indices = indices[0][1:]  # Exclude self
                
                if len(neighbor_indices) == 0:
                    continue
                
                neighbor_local_idx = np.random.choice(neighbor_indices)
                neighbor_global_idx = minority_idx[neighbor_local_idx]
                
                delta = np.random.uniform(0, 1)
                synthetic = X[global_idx] + delta * (X[neighbor_global_idx] - X[global_idx])
                synthetic_samples.append(synthetic)
        
        if len(synthetic_samples) == 0:
            return X, y
        
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(len(X_synthetic), minority_class)
        
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.concatenate([y, y_synthetic])
        
        return X_resampled, y_resampled


class BorderlineSMOTE:
    """Borderline-SMOTE (only oversample borderline minority samples)."""
    
    def __init__(self, k_neighbors=5, alpha=1.0, m_neighbors=10):
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.m_neighbors = m_neighbors
    
    def fit_resample(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[counts.argmin()]
        minority_size = counts.min()
        majority_size = counts.max()
        
        minority_idx = np.where(y == minority_class)[0]
        X_minority = X[minority_idx]
        
        # Find borderline minority samples (in DANGER)
        nbrs = NearestNeighbors(n_neighbors=min(self.m_neighbors, len(X)))
        nbrs.fit(X)
        
        borderline_idx = []
        for local_idx, global_idx in enumerate(minority_idx):
            distances, indices = nbrs.kneighbors(X[global_idx:global_idx+1])
            neighbor_labels = y[indices[0]]
            majority_neighbors = (neighbor_labels != minority_class).sum()
            
            # DANGER: at least half neighbors are majority
            if majority_neighbors >= self.m_neighbors / 2 and majority_neighbors < self.m_neighbors:
                borderline_idx.append(local_idx)
        
        if len(borderline_idx) == 0:
            # No borderline samples, use regular SMOTE
            return SMOTE(self.k_neighbors, self.alpha).fit_resample(X, y)
        
        X_borderline = X_minority[borderline_idx]
        
        n_synthetic = int(self.alpha * (majority_size - minority_size))
        
        nbrs_minority = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_minority)))
        nbrs_minority.fit(X_minority)
        
        synthetic_samples = []
        for _ in range(n_synthetic):
            local_idx = np.random.choice(len(borderline_idx))
            global_idx = minority_idx[borderline_idx[local_idx]]
            
            distances, indices = nbrs_minority.kneighbors(X[global_idx:global_idx+1])
            neighbor_indices = indices[0][1:]
            
            if len(neighbor_indices) == 0:
                continue
            
            neighbor_local_idx = np.random.choice(neighbor_indices)
            neighbor_global_idx = minority_idx[neighbor_local_idx]
            
            delta = np.random.uniform(0, 1)
            synthetic = X[global_idx] + delta * (X[neighbor_global_idx] - X[global_idx])
            synthetic_samples.append(synthetic)
        
        if len(synthetic_samples) == 0:
            return X, y
        
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(len(X_synthetic), minority_class)
        
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.concatenate([y, y_synthetic])
        
        return X_resampled, y_resampled


class FuzzySMOTE:
    """Fuzzy SMOTE (uses fuzzy memberships)."""
    
    def __init__(self, k_neighbors=5, alpha=1.0, theta=0.7):
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.theta = theta
    
    def fit_resample(self, X, fuzzy_labels):
        """
        X: numpy array [N, F]
        fuzzy_labels: numpy array [N, C]
        """
        num_classes = fuzzy_labels.shape[1]
        crisp_labels = fuzzy_labels.argmax(axis=1)
        
        unique, counts = np.unique(crisp_labels, return_counts=True)
        minority_class = unique[counts.argmin()]
        minority_size = counts.min()
        majority_size = counts.max()
        
        # Select minority nodes with low membership (fuzzy minority)
        class_memberships = fuzzy_labels[:, minority_class]
        minority_mask = class_memberships < self.theta
        minority_idx = np.where(minority_mask)[0]
        
        if len(minority_idx) == 0:
            return X, fuzzy_labels
        
        X_minority = X[minority_idx]
        fuzzy_minority = fuzzy_labels[minority_idx]
        
        n_synthetic = int(self.alpha * (majority_size - minority_size))
        
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_minority)))
        nbrs.fit(X_minority)
        
        synthetic_samples = []
        synthetic_labels = []
        
        for _ in range(n_synthetic):
            idx = np.random.randint(0, len(X_minority))
            distances, indices = nbrs.kneighbors(X_minority[idx:idx+1])
            neighbor_indices = indices[0][indices[0] != idx]
            
            if len(neighbor_indices) == 0:
                continue
            
            neighbor_idx = np.random.choice(neighbor_indices)
            
            delta = np.random.uniform(0, 1)
            synthetic_x = X_minority[idx] + delta * (X_minority[neighbor_idx] - X_minority[idx])
            
            # Fuzzy label aggregation
            mu_i = class_memberships[minority_idx[idx]]
            mu_j = class_memberships[minority_idx[neighbor_idx]]
            
            synthetic_y = (mu_i * fuzzy_minority[idx] + mu_j * fuzzy_minority[neighbor_idx]) / (mu_i + mu_j + 1e-8)
            
            synthetic_samples.append(synthetic_x)
            synthetic_labels.append(synthetic_y)
        
        if len(synthetic_samples) == 0:
            return X, fuzzy_labels
        
        X_synthetic = np.array(synthetic_samples)
        fuzzy_synthetic = np.array(synthetic_labels)
        
        X_resampled = np.vstack([X, X_synthetic])
        fuzzy_resampled = np.vstack([fuzzy_labels, fuzzy_synthetic])
        
        return X_resampled, fuzzy_resampled


# ============================================================================
# GRAPH-BASED METHODS
# ============================================================================

class GraphSMOTE:
    """GraphSMOTE - SMOTE adapted for graphs."""
    
    def __init__(self, embedding_dim=128, num_gnn_layers=2, alpha=1.0,
                 k_neighbors=5, dropout=0.5, device='cpu'):
        self.embedding_dim = embedding_dim
        self.num_gnn_layers = num_gnn_layers
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.dropout = dropout
        self.device = device
        self.encoder = None
        self.classifier = None
    
    def fit(self, data, labels, train_mask, val_mask=None, n_epochs=200,
            lr=0.01, weight_decay=5e-4, patience=20, verbose=True):
        
        # Step 1: Train encoder
        self.encoder = VanillaGNN(
            data.x.shape[1], self.embedding_dim, 
            labels.max().item() + 1 if labels.dim() == 1 else labels.shape[1],
            self.num_gnn_layers, self.dropout
        ).to(self.device)
        
        self.encoder.fit(data, labels, train_mask, val_mask, n_epochs,
                        lr, weight_decay, patience, self.device, verbose=False)
        
        # Step 2: Get embeddings
        self.encoder.eval()
        with torch.no_grad():
            embeddings = self.encoder.convs[0](data.x.to(self.device), data.edge_index.to(self.device))
            for conv in self.encoder.convs[1:]:
                embeddings = F.relu(embeddings)
                embeddings = conv(embeddings, data.edge_index.to(self.device))
        
        embeddings_np = embeddings.cpu().numpy()
        labels_np = labels.cpu().numpy() if labels.dim() == 1 else labels.argmax(dim=1).cpu().numpy()
        
        # Step 3: Apply SMOTE in embedding space
        smote = SMOTE(self.k_neighbors, self.alpha)
        X_train = embeddings_np[train_mask.cpu().numpy()]
        y_train = labels_np[train_mask.cpu().numpy()]
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Step 4: Create augmented graph
        num_synthetic = len(X_resampled) - len(X_train)
        
        if verbose:
            print(f"  Generated {num_synthetic} synthetic nodes")
        
        # For simplicity, we train on resampled embeddings
        X_aug_torch = torch.FloatTensor(X_resampled).to(self.device)
        y_aug_torch = torch.LongTensor(y_resampled).to(self.device)
        
        # Train classifier on augmented data
        self.classifier = nn.Linear(self.embedding_dim, labels.max().item() + 1 if labels.dim() == 1 else labels.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        
        for epoch in range(n_epochs // 2):
            self.classifier.train()
            optimizer.zero_grad()
            out = self.classifier(X_aug_torch)
            loss = F.cross_entropy(out, y_aug_torch)
            loss.backward()
            optimizer.step()
        
        return data, labels
    
    def predict(self, data, device='cpu'):
        self.encoder.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            data = data.to(device)
            embeddings = self.encoder.convs[0](data.x, data.edge_index)
            for conv in self.encoder.convs[1:]:
                embeddings = F.relu(embeddings)
                embeddings = conv(embeddings, data.edge_index)
            
            out = self.classifier(embeddings)
            fuzzy_pred = F.softmax(out, dim=1)
            crisp_pred = fuzzy_pred.argmax(dim=1)
        
        return fuzzy_pred, crisp_pred


class ImGAGN(nn.Module):
    """ImGAGN - Imbalanced Graph Attention Network (simplified version)."""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5, heads: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
    
    def fit(self, data, labels, train_mask, val_mask=None, n_epochs=200,
            lr=0.01, weight_decay=5e-4, patience=20, device='cpu', verbose=True):
        
        self.to(device)
        data = data.to(device)
        labels = labels.to(device)
        
        # Class-balanced loss
        if labels.dim() == 2:
            crisp_labels = labels.argmax(dim=1)
        else:
            crisp_labels = labels
        
        class_counts = torch.bincount(crisp_labels[train_mask])
        class_weights = 1.0 / (class_counts.float() + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            
            if labels.dim() == 2:
                probs = F.softmax(out[train_mask], dim=1)
                loss = -(labels[train_mask] * torch.log(probs + 1e-8) * class_weights.unsqueeze(0)).sum(dim=1).mean()
            else:
                loss = F.cross_entropy(out[train_mask], labels[train_mask], weight=class_weights)
            
            loss.backward()
            optimizer.step()
            
            if val_mask is not None:
                self.eval()
                with torch.no_grad():
                    val_out = self(data.x, data.edge_index)
                    if labels.dim() == 2:
                        val_probs = F.softmax(val_out[val_mask], dim=1)
                        val_loss = -(labels[val_mask] * torch.log(val_probs + 1e-8)).sum(dim=1).mean()
                    else:
                        val_loss = F.cross_entropy(val_out[val_mask], labels[val_mask])
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
    
    def predict(self, data, device='cpu'):
        self.eval()
        data = data.to(device)
        with torch.no_grad():
            out = self(data.x, data.edge_index)
            fuzzy_pred = F.softmax(out, dim=1)
            crisp_pred = fuzzy_pred.argmax(dim=1)
        return fuzzy_pred, crisp_pred


class GraphENS(nn.Module):
    """GraphENS - Graph Ensemble for imbalanced learning (simplified)."""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5, n_models: int = 3):
        super().__init__()
        self.models = nn.ModuleList([
            VanillaGNN(in_channels, hidden_channels, num_classes, num_layers, dropout)
            for _ in range(n_models)
        ])
        self.n_models = n_models
    
    def forward(self, x, edge_index):
        outputs = [model(x, edge_index) for model in self.models]
        return torch.stack(outputs).mean(dim=0)
    
    def fit(self, data, labels, train_mask, val_mask=None, n_epochs=200,
            lr=0.01, weight_decay=5e-4, patience=20, device='cpu', verbose=True):
        
        # Train each model independently
        for i, model in enumerate(self.models):
            if verbose:
                print(f"  Training ensemble model {i+1}/{self.n_models}...")
            model.fit(data, labels, train_mask, val_mask, n_epochs,
                     lr, weight_decay, patience, device, verbose=False)
    
    def predict(self, data, device='cpu'):
        self.eval()
        with torch.no_grad():
            outputs = [model.predict(data, device)[0] for model in self.models]
            fuzzy_pred = torch.stack(outputs).mean(dim=0)
            crisp_pred = fuzzy_pred.argmax(dim=1)
        return fuzzy_pred, crisp_pred
