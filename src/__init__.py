"""
Fuzzy-GraphSMOTE Package
========================

A unified framework for fuzzy imbalanced node classification on graphs.
"""

from .fuzzification import DatasetFuzzifier, create_cora_metaclasses
from .fuzzy_graphsmote import FuzzyGraphSMOTE
from .baselines import VanillaGNN, GraphSMOTE
from .data_loader import load_cora, load_musae_github, create_train_val_test_split
from .evaluation import FuzzyMetrics, print_comparison_table, geometric_mean_score

__version__ = '1.0.0'
__author__ = 'Your Name'

__all__ = [
    'DatasetFuzzifier',
    'create_cora_metaclasses',
    'FuzzyGraphSMOTE',
    'VanillaGNN',
    'GraphSMOTE',
    'load_cora',
    'load_musae_github',
    'create_train_val_test_split',
    'FuzzyMetrics',
    'print_comparison_table',
    'geometric_mean_score',
]
