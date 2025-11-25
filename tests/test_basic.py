"""Basic functionality tests for V16f-η"""
import pytest
import numpy as np
from cosmos_v16f import CosmosV16f

def test_initialization():
    cosmos = CosmosV16f()
    assert cosmos.ref_amp == 0.17
    assert cosmos.ref_period == 200

def test_breathing_period():
    cosmos = CosmosV16f(ref_amp=0.17, ref_period=200, seed=42)
    cosmos.run(10000)
    period = cosmos.breathing_period_mean()
    assert 90 < period < 110, f"Breathing period {period} out of expected range"

def test_reproducibility():
    cosmos1 = CosmosV16f(seed=42)
    cosmos1.run(1000)
    
    cosmos2 = CosmosV16f(seed=42)
    cosmos2.run(1000)
    
    assert np.allclose(cosmos1.history['kappa'], cosmos2.history['kappa'])
