"""
Breathing Function Package
==========================

A universal activation function for adaptive systems.

Basic usage:
    >>> from breathing import breath, BreathingFunction
    >>> breath(0.5)
    0.4621...
    >>> bf = BreathingFunction()
    >>> bf.update(user_activity)
    >>> bf.get_value()

Presets:
    >>> bf = BreathingFunction.calm()    # Meditation, ambient
    >>> bf = BreathingFunction.active()  # Games, interactive
    >>> bf = BreathingFunction.adaptive() # Day/night aware
"""

from .breathing import (
    BreathingFunction,
    BreathConfig,
    BreathState,
    BreathPhase,
    breath,
    breath_array,
)

__version__ = "0.1.0"
__author__ = "YAGC Project"
__all__ = [
    "BreathingFunction",
    "BreathConfig",
    "BreathState",
    "BreathPhase",
    "breath",
    "breath_array",
]
