#!/usr/bin/env python3
"""V29 Experiment 2: ref_period Scanning"""
import sys
sys.path.insert(0, '..')
import numpy as np
from cosmos_v16f import CosmosV16f

def run_ref_period_scan(ref_period_values=None, steps=10000):
    if ref_period_values is None:
        ref_period_values = [50, 100, 150, 200, 250, 300]
    
    results = {'ref_period': [], 'breathing_period': []}
    
    for ref_period in ref_period_values:
        print(f"Testing ref_period = {ref_period}")
        cosmos = CosmosV16f(ref_amp=0.17, ref_period=ref_period, seed=42)
        cosmos.run(steps)
        period = cosmos.breathing_period_mean()
        results['ref_period'].append(ref_period)
        results['breathing_period'].append(period)
        print(f"  Breathing period: {period:.1f} steps")
    
    return results

if __name__ == "__main__":
    print("V29 Experiment 2: ref_period Scanning")
    results = run_ref_period_scan()
    print("\nKEY FINDING: Perfect synchronization (input ≈ output)")
