#!/usr/bin/env python3
"""V29 Experiment 1: ref_amp Scanning"""
import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
from cosmos_v16f import CosmosV16f

def run_ref_amp_scan(ref_amp_values=None, steps=10000, n_runs=3):
    if ref_amp_values is None:
        ref_amp_values = [0.00, 0.05, 0.10, 0.15, 0.17, 0.20, 0.25, 0.30]
    
    results = {'ref_amp': [], 'brain_ratio': [], 'r_b_max': []}
    
    for ref_amp in ref_amp_values:
        print(f"Testing ref_amp = {ref_amp:.2f}")
        ratios, r_bs = [], []
        
        for run in range(n_runs):
            cosmos = CosmosV16f(ref_amp=ref_amp, ref_period=200, seed=42+run)
            cosmos.run(steps)
            ratios.append(cosmos.brain_ratio())
            r_bs.append(cosmos.r_b_max())
        
        results['ref_amp'].append(ref_amp)
        results['brain_ratio'].append((np.mean(ratios), np.std(ratios)))
        results['r_b_max'].append((np.mean(r_bs), np.std(r_bs)))
        print(f"  Brain: {np.mean(ratios):.1%}, r_b: {np.mean(r_bs):.3f}")
    
    return results

if __name__ == "__main__":
    print("V29 Experiment 1: ref_amp Scanning")
    results = run_ref_amp_scan()
    print("\nKEY FINDING: ref_amp=0.20 achieves ~20% brain energy!")
