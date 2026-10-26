"""Numerically stable C51 action ranking for frozen evaluation only."""
import torch

EVALUATION_MODE = "stable_float64_v1"


def action_scores_from_logits(logits, support):
    """Return E[Z - Z_min] in float64, for ranking actions, not TD targets.

    Cast before softmax: casting float32 probabilities cannot recover lost
    precision. Subtracting the common support minimum preserves the mathematical
    ordering and avoids cancellation between negative and positive atoms.
    Network weights/logits retain their original precision. Exact ties retain
    torch.argmax's first-index rule; no tolerance or random perturbation is used.
    """
    probabilities = torch.softmax(logits.to(torch.float64), dim=-1)
    atoms = support.to(device=logits.device, dtype=torch.float64)
    return (probabilities * (atoms - atoms[0])).sum(dim=-1)
