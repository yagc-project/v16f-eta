"""Tests for V18R theory matching"""
import pytest
import numpy as np
from cosmos_v16f import CosmosV16f

def test_brain_energy_ratio():
    """V18R predicts ~20-21% brain energy allocation"""
    cosmos = CosmosV16f(ref_amp=0.17, ref_period=200, seed=42)
    cosmos.run(10000)
    ratio = cosmos.brain_ratio()
    assert 0.19 < ratio < 0.24, f"Brain ratio {ratio:.1%} outside V18R range"

def test_r_b_near_threshold():
    """V16 r_b should approach consciousness threshold (0.85)"""
    cosmos = CosmosV16f(ref_amp=0.17, ref_period=200, seed=42)
    cosmos.run(10000)
    r_b_max = cosmos.r_b_max()
    assert r_b_max > 0.80, f"r_b max {r_b_max} below expected"

def test_ref_amp_20_percent():
    """ref_amp=0.20 should achieve 20% (V29 finding)"""
    cosmos = CosmosV16f(ref_amp=0.20, ref_period=200, seed=42)
    cosmos.run(10000)
    ratio = cosmos.brain_ratio()
    assert abs(ratio - 0.20) < 0.02, f"ref_amp=0.20 gave {ratio:.1%}, expected 20%"
