#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 03:09:45 2025

@author: felipemoret
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global configuration file for the Market Making + LOB simulation environment.

This file provides:
    - a global random seed (full reproducibility)
    - all simulation parameters (lam, mu, delta, arrival times)
    - additional placeholders for future constants
    - clean seeding for Python, NumPy, and PyTorch

Import this module in any script to ensure consistent configuration.
"""

import numpy as np
import random
import torch

# ============================================================
# 1) GLOBAL SEED FOR FULL REPRODUCIBILITY
# ============================================================

GLOBAL_SEED = 123

def seed_all(seed: int = GLOBAL_SEED) -> None:
    """Seed all RNGs for reproducibility. Call explicitly rather than at import."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Seed once at import for backward compatibility.
# Callers should call seed_all(seed) explicitly before each run.
seed_all(GLOBAL_SEED)


# ============================================================
# 2) LOB SIMULATION PARAMETERS
# ============================================================

"""
lam  : intensity of limit-order arrivals
mu   : intensity of market-order arrivals
delta: intensity of cancellations / order evaporation
"""

#Based on AMZN Santa Fe Calibration
lam = 0.06
mu = 0.1
delta = 0.02


# ============================================================
# 3) PAPER ENGINE: Santa Fe
# ============================================================
# Compatibility names for recorded call signatures; QRM is not distributed.

USE_QRM = False

# The paper uses only the Santa Fe engine.
qrm_params = None
mean_size_LO = 1
mean_size_MO = 1


# ============================================================
# 4) OPTIONAL: PLACEHOLDERS FOR FUTURE CONSTANTS
# ============================================================

# Example placeholders (commented out):
# MAX_INVENTORY = 50
# DEFAULT_TICK_SIZE = 1
# SIMULATION_LENGTH = 10000
# REWARD_INVENTORY_COEFF = 0.1


# ============================================================
# 4) PRINT CONFIG (optional)
# ============================================================

if __name__ == "__main__":
    print("=== CONFIG_MM.py loaded successfully ===")
    print(f"GLOBAL_SEED: {GLOBAL_SEED}")
    print(f"lam: {lam}")
    print(f"mu: {mu}")
    print(f"delta: {delta}")
