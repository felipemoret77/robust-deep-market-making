"""
Central color palette for the paper figures (Okabe-Ito, colorblind-safe).

One fixed color per recurring entity so the same thing has the same color
across every figure.  Analytical / reference variants reuse the entity color
with a dashed line style, so the figures also remain legible in grayscale.

    NAIVE    gray            at-best lower bound
    GLFT     blue            analytical benchmark   (misspecification -> dashed)
    PHASE_A  orange          stationary DQN
    PHASE_B  green           regime-adapted DQN
    BANDIT   reddish purple  scenario-bandit fine-tuning
    REF      dark gray       reference trajectory / within-figure benchmark
    BUY/SELL green/red       regime shading (buy-heavy / sell-heavy)
"""

NAIVE   = "#999999"
GLFT    = "#0072B2"
PHASE_A = "#E69F00"
PHASE_B = "#009E73"
BANDIT  = "#CC79A7"
REF     = "#444444"
BUY     = "green"
SELL    = "red"
