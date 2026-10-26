"""Paper regime schedule; function preserved from the original experiment."""

from typing import Callable, List
import numpy as np

REGIME_L_MIN = 10
REGIME_ALPHA = 1.5
REGIME_P_LO = 0.2
REGIME_P_HI = 0.8
REGIME_DISTRIBUTION = "exponential"
REGIME_EXP_TAU = 60.0


def make_regime_schedule(
    seed: int,
    n_mo_events: int,
    L_min: int = REGIME_L_MIN,
    alpha: float = REGIME_ALPHA,
    p_lo: float = REGIME_P_LO,
    p_hi: float = REGIME_P_HI,
    distribution: str = "pareto",
    exp_rate: float = 0.0,
) -> Callable[[int], float]:
    """
    Build a callable ``step_idx -> buy_mo_prob`` with regime-switching.

    This callable is designed to plug directly into the LOB engine's
    ``buy_mo_prob`` parameter (which accepts either a float or a
    ``Callable[[int], float]``).  Within each regime of random duration,
    the buy MO probability is fixed at a value drawn from ``U[p_lo, p_hi]``.

    The schedule is fully pre-generated at construction time (no online
    randomness), which ensures reproducibility given the same seed and
    allows efficient O(log n) lookup per call via ``np.searchsorted``.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility of the regime schedule.
    n_mo_events : int
        Expected number of MO events per episode.  Used to pre-generate
        enough regime boundaries to cover the full episode (with
        a generous +1000 buffer for stochastic variation in MO counts).
    L_min : int
        Minimum regime duration (Pareto scale parameter), in MO events.
        With ``alpha=1.5``, the expected regime length is ``3 * L_min``.
        Ignored when ``distribution="exponential"``.
    alpha : float
        Pareto shape parameter.  Controls the tail heaviness of regime
        durations.  ``alpha=1.5`` produces heavy tails consistent with
        empirical metaorder durations (Lillo, Mike & Farmer 2005).
        Larger alpha -> lighter tails, more uniform regime lengths.
        Ignored when ``distribution="exponential"``.
    p_lo, p_hi : float
        Bounds for the uniform draw of p_buy within each regime.
        Default [0.2, 0.8] is symmetric around 0.5 to preserve the
        global mean flow direction while creating local asymmetry.
    distribution : str
        ``"pareto"`` (default) or ``"exponential"``.
        When ``"exponential"``, regime durations are drawn from
        ``Exp(exp_rate)`` (mean = 1/exp_rate MO events).
    exp_rate : float
        Rate parameter λ for exponential regime durations (λ = 1/τ).
        Required when ``distribution="exponential"``.
        The optimal EWMA alpha (time constant) is ``1 - exp(-λ)``.

    Returns
    -------
    schedule : Callable[[int], float]
        Maps MO event index -> buy_mo_prob for that event.
        Also exposes ``.boundaries`` (list of regime boundary indices)
        and ``.p_values`` (list of p_buy per regime) for diagnostics.
    """

    if distribution not in ("pareto", "exponential"):
        raise ValueError(
            f"distribution must be 'pareto' or 'exponential', got '{distribution}'"
        )
    if distribution == "exponential" and exp_rate <= 0.0:
        raise ValueError(
            f"exp_rate must be > 0 for exponential distribution, got {exp_rate}"
        )

    rng = np.random.default_rng(seed)

    # Pre-generate regime boundaries and p_buy values for the expected
    # horizon. If a caller later queries beyond that horizon, we extend the
    # schedule on demand instead of freezing at the last p_buy forever.
    boundaries: List[int] = [0]
    p_values: List[float] = []
    boundaries_arr = np.array(boundaries)
    p_values_arr = np.array(p_values)

    def _draw_length() -> int:
        if distribution == "pareto":
            # Pareto draw via inverse CDF: L = ceil(L_min / U^(1/alpha))
            return int(np.ceil(L_min / rng.uniform() ** (1.0 / alpha)))
        # Exponential draw: L = ceil(Exp(1/rate))
        return max(1, int(np.ceil(rng.exponential(1.0 / exp_rate))))

    def _append_regime() -> None:
        nonlocal boundaries_arr, p_values_arr
        L = _draw_length()
        boundaries.append(boundaries[-1] + L)
        p_values.append(float(rng.uniform(p_lo, p_hi)))
        boundaries_arr = np.array(boundaries)
        p_values_arr = np.array(p_values)

    while boundaries[-1] < n_mo_events + 1000:
        _append_regime()

    boundaries_arr = np.array(boundaries)
    p_values_arr = np.array(p_values)

    def schedule(step_idx: int) -> float:
        """O(log n) lookup: extend if needed, then return the current regime p_buy."""
        while step_idx >= boundaries[-1]:
            _append_regime()
        regime_idx = int(np.searchsorted(boundaries_arr, step_idx, side="right")) - 1
        regime_idx = min(regime_idx, len(p_values_arr) - 1)
        return p_values_arr[regime_idx]

    # Attach metadata for diagnostics and plotting
    schedule.boundaries = boundaries  # type: ignore[attr-defined]
    schedule.p_values = p_values  # type: ignore[attr-defined]

    return schedule
