#!/usr/bin/env python3
"""
V16f-η: The Breathing Function
================================

Computational implementation of the breathing function foundation for
YAGC's time-centric quantum gravitational cosmology.

Author: YAGC Project
License: CC BY 4.0
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

__version__ = "1.0.0"

@dataclass
class CosmosState:
    """State variables of the breathing function."""
    kappa: float = 0.5
    kappa_dot: float = 0.0
    r_t: float = 0.05
    r_g: float = 0.27
    r_b: float = 0.68
    T: float = 0.25
    coherence: float = 0.7
    pi_integral: float = 0.0
    kappa_dot_prev: float = 0.0
    last_zero_cross: int = 0
    breathing_period: float = 100.0

class CosmosV16f:
    """V16f-η Breathing Function Implementation"""
    
    def __init__(self, ref_amp=0.17, ref_period=200, seed=None, **kwargs):
        # Core parameters
        self.ref_amp = ref_amp
        self.ref_period = ref_period
        self.mu = kwargs.get('mu', 0.68)
        self.omega0 = kwargs.get('omega0', 0.09)
        self.kappa_speed = kwargs.get('kappa_speed', 0.11)
        self.breath_inertia_beta = kwargs.get('breath_inertia_beta', 0.76)
        self.pi_kp = kwargs.get('pi_kp', 0.09)
        self.pi_ki = kwargs.get('pi_ki', 0.005)
        self.coh_cap = kwargs.get('coh_cap', 0.9)
        self.T_min = kwargs.get('T_min', 0.1)
        self.T_max = kwargs.get('T_max', 0.4)
        self.chaos_mix = kwargs.get('chaos_mix', 0.5)
        
        self.state = CosmosState()
        self.history = {k: [] for k in ['kappa', 'kappa_dot', 'r_t', 'r_g', 'r_b', 'T', 'coherence', 'breathing_period']}
        self.target_r = np.array([0.05, 0.27, 0.68])
        
        if seed is not None:
            np.random.seed(seed)
            
    def reference_signal(self, t):
        return self.ref_amp * np.sin(2 * np.pi * t / self.ref_period)
    
    def tar_softmax(self, kappa, T, coherence, lock, pi_err):
        L0 = np.log(self.target_r)
        L0 -= L0.mean()
        
        bias_t = -0.5 * kappa + 0.3 * T - 0.2 * coherence
        bias_g = 0.3 * kappa - 0.1 * T + 0.1 * coherence
        bias_b = 1.2 * kappa - 0.2 * (T - 0.2) + 0.15 * lock
        
        L = L0 + np.array([bias_t, bias_g, bias_b]) + np.array([0.0, 0.0, pi_err])
        exp_L = np.exp(L - L.max())
        r = exp_L / exp_L.sum()
        return r[0], r[1], r[2]
    
    def step(self, t):
        S = self.state
        k_ref = self.reference_signal(t)
        
        k_ddot = self.mu * (1.0 - S.kappa**2) * S.kappa_dot - self.omega0**2 * (S.kappa - k_ref)
        k_ddot += self.chaos_mix * 0.001 * np.random.randn()
        
        S.kappa_dot = self.breath_inertia_beta * S.kappa_dot_prev + (1 - self.breath_inertia_beta) * k_ddot
        
        if S.kappa < 0.08:
            S.kappa_dot += 0.005
        if S.kappa > 0.97:
            S.kappa_dot -= 0.002
            
        S.kappa += S.kappa_dot * self.kappa_speed
        S.kappa = np.clip(S.kappa, 0.0, 1.0)
        
        if S.kappa_dot * S.kappa_dot_prev < 0:
            S.breathing_period = t - S.last_zero_cross
            S.last_zero_cross = t
            
        S.kappa_dot_prev = S.kappa_dot
        
        lock = 1.0 if S.r_b > 0.75 else 0.0
        coh_update = 0.62 * lock + 0.26 * S.kappa - 0.18 * S.T + 0.34 * S.coherence
        S.coherence = np.clip(coh_update, 0.0, self.coh_cap)
        
        T_update = S.T * 0.95 + 0.05 * (1.0 - S.r_b) + 0.02 * (1.0 - S.coherence)
        S.T = np.clip(T_update, self.T_min, self.T_max)
        
        pi_error = self.target_r[2] - S.r_b
        S.pi_integral += pi_error
        S.pi_integral = np.clip(S.pi_integral, -10.0, 10.0)
        pi_output = self.pi_kp * pi_error + self.pi_ki * S.pi_integral
        
        S.r_t, S.r_g, S.r_b = self.tar_softmax(S.kappa, S.T, S.coherence, lock, pi_output)
        
    def run(self, steps, store_every=1):
        for t in range(steps):
            self.step(t)
            if t % store_every == 0:
                for key in self.history:
                    self.history[key].append(getattr(self.state, key))
                    
    def brain_ratio(self, window=-5000):
        if len(self.history['r_b']) == 0:
            return 0.0
        r_b_arr = np.array(self.history['r_b'][window:])
        r_t_arr = np.array(self.history['r_t'][window:])
        return r_t_arr.mean() + (1 - r_b_arr.mean()) * 0.5
    
    def r_b_max(self, window=-5000):
        if len(self.history['r_b']) == 0:
            return 0.0
        return np.max(self.history['r_b'][window:])
    
    def breathing_period_mean(self, window=-5000):
        if len(self.history['breathing_period']) == 0:
            return 0.0
        periods = np.array(self.history['breathing_period'][window:])
        valid = periods[periods > 0]
        return valid.mean() if len(valid) > 0 else 0.0
    
    def plot_summary(self, figsize=(14, 10)):
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        time = np.arange(len(self.history['kappa']))
        
        axes[0, 0].plot(time, self.history['kappa'], 'b-', lw=0.5)
        axes[0, 0].set_ylabel('κ', fontsize=12)
        axes[0, 0].set_title('van der Pol Oscillator', fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].plot(time, self.history['r_t'], 'g-', label='r_t', lw=0.8)
        axes[0, 1].plot(time, self.history['r_g'], 'orange', label='r_g', lw=0.8)
        axes[0, 1].plot(time, self.history['r_b'], 'b-', label='r_b', lw=0.8)
        axes[0, 1].axhline(0.85, color='r', ls='--', alpha=0.5, label='Threshold')
        axes[0, 1].legend()
        axes[0, 1].set_title('TAR-Softmax', fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].plot(time, self.history['coherence'], 'purple', lw=0.8)
        axes[1, 0].set_ylabel('Coherence')
        axes[1, 0].set_title('Temporal Coherence', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].plot(time, self.history['T'], 'r-', lw=0.8)
        axes[1, 1].set_ylabel('Temperature')
        axes[1, 1].set_title('Entropy', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        axes[2, 0].plot(self.history['kappa'], self.history['r_b'], 'b-', alpha=0.3, lw=0.3)
        axes[2, 0].set_xlabel('κ')
        axes[2, 0].set_ylabel('r_b')
        axes[2, 0].set_title('Phase Space', fontweight='bold')
        axes[2, 0].grid(alpha=0.3)
        
        stats = f"Brain ratio: {self.brain_ratio():.1%}\nr_b max: {self.r_b_max():.3f}\nPeriod: {self.breathing_period_mean():.1f} steps"
        axes[2, 1].text(0.1, 0.5, stats, fontsize=11, family='monospace',
                       bbox=dict(boxstyle='round', fc='wheat', alpha=0.3))
        axes[2, 1].axis('off')
        axes[2, 1].set_title('Statistics', fontweight='bold')
        
        plt.tight_layout()
        return fig

def run_simulation(steps=10000, ref_amp=0.17, ref_period=200, seed=None, plot=True):
    cosmos = CosmosV16f(ref_amp=ref_amp, ref_period=ref_period, seed=seed)
    cosmos.run(steps)
    print(f"Brain energy ratio: {cosmos.brain_ratio():.1%}")
    print(f"r_b max: {cosmos.r_b_max():.3f}")
    print(f"Period: {cosmos.breathing_period_mean():.1f} steps")
    if plot:
        cosmos.plot_summary()
        plt.show()
    return cosmos

if __name__ == "__main__":
    print("V16f-η Breathing Function")
    print("=" * 50)
    cosmos = run_simulation(steps=10000, seed=42)
