"""Reproducibility tests"""
import pytest
import numpy as np
from cosmos_v16f import CosmosV16f

def test_seed_reproducibility():
    """Same seed should give identical results"""
    cosmos1 = CosmosV16f(seed=42)
    cosmos1.run(1000)
    
    cosmos2 = CosmosV16f(seed=42)
    cosmos2.run(1000)
    
    for key in ['kappa', 'r_b', 'coherence']:
        assert np.allclose(cosmos1.history[key], cosmos2.history[key])

def test_different_seeds_differ():
    """Different seeds should give different results"""
    cosmos1 = CosmosV16f(seed=42)
    cosmos1.run(1000)
    
    cosmos2 = CosmosV16f(seed=123)
    cosmos2.run(1000)
    
    assert not np.allclose(cosmos1.history['kappa'], cosmos2.history['kappa'])
