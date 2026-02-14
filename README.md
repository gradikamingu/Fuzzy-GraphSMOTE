# 🎯 Fuzzy-GraphSMOTE - Official Implementation

**Final version for the article** | **Requirements**: Python 3.11 | PyTorch 2.1.0

---

## EXPLORATION (approx. 30 min)

```powershell
python 01_explore_cora.py      # Data exploration on Cora dataset
```

## EXPERIMENTS (30-180 min depending on the config)

```powershell
# Quick test run
python 02_run_experiments.py --datasets cora --methods vanilla_gnn fuzzy_graphsmote --n_runs 1

# Full experiment run
python 02_run_experiments.py --datasets cora musae --methods vanilla_gnn gcn graphsmote fuzzy_graphsmote --n_runs 5
```

## COMPARED BASELINES

1. **Vanilla GNN** (GraphSAGE) - without oversampling
2. **GCN** - Graph Convolutional Network
3. **GraphSMOTE** - Oversampling graph-aware (crisp)
4. **SMOTE** - Oversampling matrix-space (crisp)
5. **Fuzzy-SMOTE** - Oversampling matrix-space (fuzzy)
6. **Fuzzy-GraphSMOTE** ⭐ (fuzzy + graph)


## METRICS EVALUATED

- Accuracy, Balanced Accuracy
- Macro F1, Precision, Recall
- **G-Mean** (crucial for imbalance)
- Fuzzy Accuracy, Fuzzy Cross-Entropy


## IMPLEMENTED METHODOLOGY

### The 8 Steps of Fuzzy-GraphSMOTE

1. **Node Embeddings**: Generates latent representations using GraphSAGE.
2. **Fuzzy Minority Identification**: Detects nodes with membership μ < θ.
3. **Synthetic Node Generation**: Performs fuzzy interpolation within the latent space.
4. **Fuzzy Edge Generation**: Predicts connections using a trainable interaction matrix S.
5. **Augmented Graph Construction**: Integrates synthetic nodes and predicted edges.
6. **Fuzzy GNN Classifier**: Training via customized fuzzy cross-entropy loss.
7. **Joint Optimization**: Two-stage process (Pre-training + Joint Optimization).
8. **Prediction**: Final inference of fuzzy memberships for node classification.

### Fuzzification

**Garanties**:
- ✅ 95% 95% of nodes: μ_dominant ≥ 0.6
- ✅ 5% 95% of nodes: Ambiguous (balanced neighborhoods)

## CONFIGURATION

### Default Hyperparameters

```python
embedding_dim = 128
alpha = 1.0          # Oversampling factor
theta = 0.7          # Minority threshold
k_neighbors = 5      # Neighbors for SMOTE interpolation
lambda_edge = 0.1    # Edge reconstruction loss weight
n_pre_epochs = 50    # Pre-training phase
n_joint_epochs = 150 # Joint optimization phase
```

### Recommended Tuning

**High Imbalance**: `alpha=2.0, theta=0.6`  
**Large-scale Graphs**: `embedding_dim=64, k_neighbors=3`  
**Overfitting**: `dropout=0.7, weight_decay=1e-3`


## CITATION

If you find this code useful for your research, please cite:

```bibtex
@article{fuzzy_graphsmote_2025,
  title={Fuzzy-GraphSMOTE: A Fuzzy Graph-based Approach for 
         Imbalanced Node Classification with Label Uncertainty},
  author={Votre Nom},
  journal={Votre Journal},
  year={2025}
}
```
