# API Reference

## CosmosV16f

Main class implementing the breathing function.

### Constructor

```python
CosmosV16f(
    ref_amp=0.17,          # External pacing amplitude
    ref_period=200,        # External pacing period
    mu=0.68,               # van der Pol damping
    omega0=0.09,           # Natural frequency
    kappa_speed=0.11,      # Time scaling
    breath_inertia_beta=0.76,  # Inertia filter
    seed=None              # Random seed
)
```

### Methods

#### run(steps, store_every=1)
Run simulation for specified number of steps.

**Parameters:**
- `steps` (int): Number of time steps
- `store_every` (int): Store history every N steps

#### brain_ratio(window=-5000)
Compute brain energy allocation ratio.

**Returns:** float - Brain energy ratio (typically ~0.215)

#### r_b_max(window=-5000)
Compute maximum information integrity.

**Returns:** float - Maximum r_b value

#### breathing_period_mean(window=-5000)
Compute mean breathing period.

**Returns:** float - Mean period in steps

#### plot_summary(figsize=(14, 10))
Create comprehensive summary plot.

**Returns:** matplotlib Figure object

## Functions

### run_simulation(steps=10000, ref_amp=0.17, ref_period=200, seed=None, plot=True)

Convenience function to run a standard simulation.

**Returns:** CosmosV16f object

## Example

```python
from cosmos_v16f import CosmosV16f

# Create instance
cosmos = CosmosV16f(ref_amp=0.17, ref_period=200, seed=42)

# Run simulation
cosmos.run(10000)

# Get results
print(f"Brain ratio: {cosmos.brain_ratio():.1%}")
print(f"r_b max: {cosmos.r_b_max():.3f}")

# Plot
cosmos.plot_summary()
```
