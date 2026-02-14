"""
PRÉ-VÉRIFICATION COMPLÈTE
Vérifie que tout fonctionne avant de lancer les expériences de 6-8h
"""

import sys
import os

print("="*80)
print("PRÉ-VÉRIFICATION COMPLÈTE")
print("="*80)

errors = []
warnings = []

# 1. Vérification des imports
print("\n[1/6] Vérification des imports Python...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except:
    errors.append("PyTorch non installé")

try:
    import torch.nn.functional as F
    print("  ✓ torch.nn.functional")
except:
    errors.append("torch.nn.functional import failed")

try:
    from torch_geometric.data import Data
    import torch_geometric
    print(f"  ✓ PyTorch Geometric")
except:
    errors.append("PyTorch Geometric non installé")

try:
    import pandas
    print(f"  ✓ Pandas {pandas.__version__}")
except:
    warnings.append("Pandas non installé (optionnel)")

try:
    import matplotlib
    print(f"  ✓ Matplotlib {matplotlib.__version__}")
except:
    warnings.append("Matplotlib non installé (optionnel)")

# 2. Vérification des fichiers du projet
print("\n[2/6] Vérification des fichiers du projet...")
required_files = [
    'src/data_loader.py',
    'src/citeseer_loader.py',
    'src/fuzzy_graphsmote.py',
    'src/fuzzification.py',
    'src/baselines.py',
    'src/advanced_baselines.py',
    'src/evaluation.py',
    '04_run_experiments.py',
    'final_experiments.py',
    'run_ablation_study.py'
]

for file in required_files:
    if os.path.exists(file):
        print(f"  ✓ {file}")
    else:
        errors.append(f"Fichier manquant: {file}")

# 3. Vérification CiteSeer
print("\n[3/6] Vérification dataset CiteSeer...")
citeseer_files = [
    'data/citeseer/citeseer.content',
    'data/citeseer/citeseer.cites'
]

citeseer_ok = True
for file in citeseer_files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024  # KB
        print(f"  ✓ {file} ({size:.1f} KB)")
    else:
        errors.append(f"CiteSeer manquant: {file}")
        citeseer_ok = False

if not citeseer_ok:
    print("\n  ⚠️  CITESEER NON TROUVÉ!")
    print("  Téléchargez depuis:")
    print("  https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.zip")
    print("  Extrayez dans: data/citeseer/")

# 4. Test chargement Cora
print("\n[4/6] Test chargement Cora...")
try:
    sys.path.append('./src')
    from data_loader import load_cora
    data, fuzzy, crisp = load_cora(alpha=0.4, seed=42)
    print(f"  ✓ Cora chargé: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"  ✓ Features: {data.x.shape[1]} dimensions")
    print(f"  ✓ Classes: {crisp.unique().tolist()}")
except Exception as e:
    errors.append(f"Erreur chargement Cora: {e}")

# 5. Test chargement CiteSeer
print("\n[5/6] Test chargement CiteSeer...")
if citeseer_ok:
    try:
        from citeseer_loader import load_citeseer
        data, fuzzy, crisp = load_citeseer(alpha=0.4, seed=42)
        print(f"  ✓ CiteSeer chargé: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"  ✓ Features: {data.x.shape[1]} dimensions")
        print(f"  ✓ Classes: {crisp.unique().tolist()}")
    except Exception as e:
        errors.append(f"Erreur chargement CiteSeer: {e}")
else:
    warnings.append("CiteSeer non testé (fichiers manquants)")

# 6. Test imports des baselines
print("\n[6/6] Test des baselines...")
try:
    from baselines import VanillaGNN, GCN, GraphSMOTE, SMOTE, FuzzySMOTE
    print("  ✓ VanillaGNN, GCN, GraphSMOTE, SMOTE, FuzzySMOTE")
except Exception as e:
    errors.append(f"Erreur import baselines: {e}")

try:
    from advanced_baselines import GATSMOTE, GraphMixup, GraphENS
    print("  ✓ GATSMOTE, GraphMixup, GraphENS")
except Exception as e:
    errors.append(f"Erreur import advanced_baselines: {e}")

try:
    from fuzzy_graphsmote import FuzzyGraphSMOTE
    print("  ✓ FuzzyGraphSMOTE")
except Exception as e:
    errors.append(f"Erreur import FuzzyGraphSMOTE: {e}")

# Résumé
print("\n" + "="*80)
if len(errors) == 0:
    print("✓✓✓ TOUTES LES VÉRIFICATIONS RÉUSSIES! ✓✓✓")
    print("="*80)
    print("\nVous pouvez lancer les expériences:")
    print("\n  python final_experiments.py")
    print("\nOu manuellement:")
    print("\n  # Cora + CiteSeer (6-8h)")
    print("  python 04_run_experiments.py --datasets cora citeseer --methods vanilla_gnn gcn graphsmote smote fuzzy_smote gatsmote graphmixup graphens fuzzy_graphsmote --n_runs 5")
    print("\n  # Juste Cora (2-3h)")
    print("  python 04_run_experiments.py --datasets cora --methods vanilla_gnn gcn graphsmote smote fuzzy_smote gatsmote graphmixup graphens fuzzy_graphsmote --n_runs 5")
    print("\n  # Juste CiteSeer (3-4h)")
    print("  python 04_run_experiments.py --datasets citeseer --methods vanilla_gnn gcn graphsmote smote fuzzy_smote gatsmote graphmixup graphens fuzzy_graphsmote --n_runs 5")
else:
    print("✗✗✗ ERREURS DÉTECTÉES ✗✗✗")
    print("="*80)
    for i, err in enumerate(errors, 1):
        print(f"\n{i}. {err}")
    
    print("\n" + "="*80)
    print("ACTIONS REQUISES:")
    print("="*80)
    
    if "PyTorch" in str(errors):
        print("\n1. Installer PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    
    if "Geometric" in str(errors):
        print("\n2. Installer PyTorch Geometric:")
        print("   pip install torch-geometric")
        print("   pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html")
    
    if "CiteSeer" in str(errors):
        print("\n3. Télécharger CiteSeer:")
        print("   URL: https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.zip")
        print("   Extraire dans: data/citeseer/")
    
    sys.exit(1)

if len(warnings) > 0:
    print("\n⚠️  AVERTISSEMENTS (non-bloquants):")
    for warn in warnings:
        print(f"  - {warn}")

print("="*80)
