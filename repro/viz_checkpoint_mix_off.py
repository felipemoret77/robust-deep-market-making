"""
viz_checkpoint_mix_off.py
=========================
Carrega o checkpoint `bayes_on2_mix_off_final` (11/05) APENAS para gerar os
plots no padrao do DeepSarsaQRunner_REGIME.py:

  (A) Funcoes de VALOR  -> heatmaps 2D + superficies 3D de V(s)=max_a Q(s,a)
      (copiadas verbatim do Runner; so troquei plt.show() por fig.savefig()).
  (B) INVENTARIO + PnL TOTAL -> rollouts greedy de avaliacao via
      simulate_LOB_with_MM, plotando MM_Inventory e MM_TotalPnL.

RODAR NO MAC (precisa de torch — a VM Windows nao tem). NAO foi executado aqui;
pontos de risco marcados com  # RISCO:.

Uso:
    cd /Users/felipemoret/Desktop/MM_LOB_SIM
    python viz_checkpoint_mix_off.py
    python viz_checkpoint_mix_off.py final --queue-mode paper --no-sim
    python viz_checkpoint_mix_off.py final --base rollout --queue-mode runner --no-sim
    python viz_checkpoint_mix_off.py final --queue-min 1 --queue-max 4 --no-sim
"""

import os
import sys
import re
import copy
import math
import random
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless: salva em arquivo, nao abre janela
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401  (registra a projecao '3d')

# === Paper figure style: matches GLFT_studies.py / risk_return_frontier.py
# / comparison_*_by_throttle.py.  Viridis cycler for multi-line plots. ===
plt.rcParams.update({
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "legend.framealpha": 0.9,
    "legend.fancybox": True,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.6,
})
from cycler import cycler as _cycler
_VIRIDIS_PAPER_PALETTE = [
    tuple(c) for c in plt.cm.viridis(np.linspace(0.05, 0.95, 6))
]
plt.rcParams["axes.prop_cycle"] = _cycler(color=_VIRIDIS_PAPER_PALETTE)

import torch

# Garante que os modulos do projeto resolvam quando rodado de dentro da pasta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# CONFIG  (caminhos do MAC)
# =====================================================================
CKPT_DIR = "/Users/felipemoret/Desktop/MM_LOB_SIM/checkpoints"
VIZ_ROOT = "/Users/felipemoret/Desktop/MM_LOB_SIM"
_CKPT_STEM = (
    "deep_mm_mtm_pure_invp0.0010_g0p999_nstep3_h1x256_a6_regime_exponential_"
    "factored_noise_fully_noisy_inv_wall_dampened_reward_cyc_lr_fill_quote_"
    "exposure_bayes_on2_mix_off"
)
# atalhos: 'final' e 'best' = os dois snapshots do MESMO run (11/05), mesma config
KNOWN_CKPTS = {
    "final": os.path.join(CKPT_DIR, _CKPT_STEM + "_final.pt"),
    "best":  os.path.join(CKPT_DIR, _CKPT_STEM + "_best_ma100.pt"),
}
# (re)definidos em main() a partir dos argumentos de linha de comando.
# DEFAULT = final (snapshot do fim do run, 11/05 13:04).
CKPT = KNOWN_CKPTS["final"]
OUT_DIR = os.path.join(VIZ_ROOT, "viz_mix_off_final")

# Estado nominal fixo para os plots de valor (features nao-varridas ficam aqui).
# Mexer aqui muda o NIVEL das superficies; o FORMATO vem das features varridas.
BASE_STATE_NOMINAL: Dict[str, Any] = {
    "spread": 2.0,
    "inventory": 0.0,
    "pure_mm_bid_sizes": [5.0, 5.0],   # offsets 0,1  (K+1 = 2)
    "pure_mm_ask_sizes": [5.0, 5.0],
    "bayes_m_hat": 0.0,                # vies direcional neutro
    "bayes_expected_run_length": 0.7,      # regime neutro, ja estabelecido
    "fill_imbalance_ewma": 0.0,        # quote-exposure neutro
}

N_EVAL_EPISODES = 20                   # rollouts para as bandas de inv/PnL
N_STEPS = 5000
N_STEPS_TO_EQUIL = 1000
INV_LIMIT = 8
BUY_MO_PROB = 0.5                      # estacionario; ver nota de REGIME no fim

# Seed do ULTIMO episodio de treino do final.pt (a mm_df dos plots veio dele):
#   current_ep_seed = GLOBAL_SEED(123) + 1000 + ep(3999) = 5122
# (reseta numpy/random/torch no inicio do episodio -> controla env E ruido NoisyNet)
LAST_EP_SEED = 5122


def _slug(title: str) -> str:
    s = title.lower().replace("/", "_").replace("+", "p")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _out_dir_for(ckpt_path: str) -> str:
    """Subpasta de saida derivada do nome do checkpoint (best vs final nao colidem)."""
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    snap = ("best_ma100" if "best_ma100" in stem
            else "final" if stem.endswith("final") else "ckpt")
    mix = "mix_on" if "mix_on" in stem else "mix_off" if "mix_off" in stem else ""
    tag = "_".join(x for x in [mix, snap] if x) or "ckpt"
    return os.path.join(VIZ_ROOT, "viz_" + tag)


def _queue_out_tag(args) -> str:
    if args.queue_min is not None or args.queue_max is not None:
        q_min = 1.0 if args.queue_min is None else float(args.queue_min)
        q_max = 4.0 if args.queue_max is None else float(args.queue_max)
        return f"queue{q_min:g}_{q_max:g}".replace(".", "p")
    if args.queue_mode == "paper":
        return "queue1_4"
    if args.queue_mode == "runner":
        return "queue_runner"
    if args.queue_mode == "wide":
        return "queue_wide"
    return "queue_auto"


# =====================================================================
# BASE_STATE via rollout — replica a metodologia do Runner
# (_last_valid_rl_state_row + _decode_pure_mm_network_vector). Helpers
# extraidos verbatim do DeepSarsaQRunner_REGIME.py (deep_controller como arg).
# =====================================================================
def _finite_int_or_none(x):
    try:
        if x is None:
            return None
        xf = float(x)
        return int(xf) if np.isfinite(xf) else None
    except (TypeError, ValueError):
        return None


def _last_valid_rl_state_row(mm_df, mode="pure_mm"):
    required = ["MM_RL_State_Mode", "MM_RL_State_Vector"]
    if not all(c in mm_df.columns for c in required):
        return None
    for idx in reversed(mm_df.index):
        row = mm_df.loc[idx]
        vec = row.get("MM_RL_State_Vector", None)
        if row.get("MM_RL_State_Mode", None) == mode and isinstance(vec, (list, np.ndarray)) and len(vec) > 0:
            return row
    return None


def _decode_pure_mm_network_vector(vec, k_offset, deep_controller):
    vec = list(vec)
    k_offset = int(k_offset)
    depth_len = k_offset + 1
    base_len = 2 + 2 * depth_len
    if len(vec) < base_len:
        raise ValueError(f"vetor pure_mm curto: len={len(vec)}, esperado>={base_len}")

    def _inv_log1p(x):
        return float(np.expm1(max(0.0, float(x))))

    inv_denom = float(deep_controller.inv_limit) if deep_controller.inv_limit is not None else 10.0
    inv_denom = max(inv_denom, 1.0)
    state = {
        "spread": _inv_log1p(vec[0]),
        "inventory": float(vec[1]) * inv_denom,
        "pure_mm_bid_sizes": [_inv_log1p(x) for x in vec[2:2 + depth_len]],
        "pure_mm_ask_sizes": [_inv_log1p(x) for x in vec[2 + depth_len:2 + 2 * depth_len]],
    }
    cursor = base_len
    if getattr(deep_controller, "use_flow_signal", False) and cursor < len(vec):
        state["mo_flow_p_hat"] = float(vec[cursor]); cursor += 1
    if getattr(deep_controller, "use_fast_flow_signal", False) and cursor < len(vec):
        state["mo_flow_fast_p_hat"] = float(vec[cursor]); cursor += 1
    if getattr(deep_controller, "use_bayes_flow_signal", False):
        bayes_keys = tuple(getattr(deep_controller, "bayes_flow_feature_keys",
                                   ("bayes_m_hat", "bayes_cp_prob",
                                    "bayes_expected_run_length", "bayes_uncertainty")))
        for key in bayes_keys:
            if cursor < len(vec):
                state[key] = float(vec[cursor]); cursor += 1
    if getattr(deep_controller, "use_fill_imbalance", False) and cursor < len(vec):
        state["fill_imbalance_ewma"] = float(vec[cursor])
    return state


def _attach_flow_features_from_row(base_state, row, deep_controller):
    if getattr(deep_controller, "use_bayes_flow_signal", False):
        col_map = {
            "bayes_m_hat": "MM_Bayes_M_Hat",
            "bayes_cp_prob": "MM_Bayes_CP_Prob",
            "bayes_expected_run_length": "MM_Bayes_Run_Length_Feature",
            "bayes_uncertainty": "MM_Bayes_Uncertainty",
        }
        for key, col in col_map.items():
            if key not in base_state and col in row:
                base_state[key] = float(row[col])
    if getattr(deep_controller, "use_fill_imbalance", False) and "fill_imbalance_ewma" not in base_state:
        for col in ("MM_Fill_Imbalance_Feature", "MM_Fill_Imbalance_EWMA"):
            if col in row:
                base_state["fill_imbalance_ewma"] = float(row[col]); break
    return base_state


def derive_base_state_from_rollout(ctrl, seed=None):
    """Roda 1 episodio greedy e usa o ULTIMO estado RL valido como base_state
    (como o Runner fazia com o mm_df do fim do treino). seed=LAST_EP_SEED por
    padrao -> replica o ENV do ultimo episodio de treino (ep 3999).
    Tambem devolve a grade runner-style das filas L1: percentis 5%-95% das
    filas realmente visitadas no episodio, como no Runner original."""
    from MM_LOB_SIM import simulate_LOB_with_MM
    from CONFIG_MM import lam, mu, delta, mean_size_LO
    try:
        from CONFIG_MM import USE_QRM, qrm_params
    except Exception:
        USE_QRM, qrm_params = False, {}

    seed = int(LAST_EP_SEED if seed is None else seed)
    print(f"[A] rollout p/ base_state com seed={seed} "
          f"(={'ultimo ep de treino' if seed == LAST_EP_SEED else 'custom'})")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sk = dict(
        lam=lam, mu=mu, delta=delta,
        number_tick_levels=50, n_priority_ranks=100, number_levels_to_store=20,
        p0=100, mean_size_LO=mean_size_LO,
        iterations=N_STEPS, iterations_to_equilibrium=N_STEPS_TO_EQUIL,
        path_save_files=None, label_simulation=None,
        controller=ctrl, random_seed=seed, buy_mo_prob=BUY_MO_PROB,
    )
    if USE_QRM:
        sk["qrm_params"] = qrm_params
    _, _, mm_df = simulate_LOB_with_MM(**sk)
    row = _last_valid_rl_state_row(mm_df, "pure_mm")
    if row is None:
        raise RuntimeError("mm_df sem MM_RL_State_Vector/Mode pure_mm valido")
    K = _finite_int_or_none(row.get("MM_RL_State_MaxOffset"))
    if K is None:
        K = int(getattr(ctrl, "max_offset", 1))
    base = _decode_pure_mm_network_vector(row["MM_RL_State_Vector"], K, ctrl)
    base = _attach_flow_features_from_row(base, row, ctrl)

    q0_all = []
    if "MM_RL_State_Vector" in mm_df.columns and "MM_RL_State_Mode" in mm_df.columns:
        for vec, mode in zip(mm_df["MM_RL_State_Vector"], mm_df["MM_RL_State_Mode"]):
            if mode != "pure_mm" or vec is None or not isinstance(vec, (list, np.ndarray)):
                continue
            try:
                decoded_row = _decode_pure_mm_network_vector(vec, K, ctrl)
            except (TypeError, ValueError):
                continue
            q0_all.append(float(decoded_row["pure_mm_bid_sizes"][0]))
            q0_all.append(float(decoded_row["pure_mm_ask_sizes"][0]))

    queue_values = None
    if q0_all:
        q0_all = np.asarray(q0_all, dtype=float)
        q_min = max(0.0, float(np.percentile(q0_all, 5)))
        q_max = float(np.percentile(q0_all, 95))
        if q_max <= q_min:
            q_max = q_min + 1.0
        queue_values = np.linspace(q_min, q_max, num=35)

    return base, queue_values


# =====================================================================
# (A) FUNCOES DE VALOR  — copiadas VERBATIM do DeepSarsaQRunner_REGIME.py
#     unica mudanca: plt.show() -> fig.savefig()+plt.close() nos plotters.
# =====================================================================
def plot_value_slices_2d(
    deep_controller,
    base_state: Dict[str, Any],
    inv_values,
    spread_values,
    bid0_values=None,
    ask0_values=None,
    bid_size_values=None,
    ask_size_values=None,
    bayes_m_values=None,
    bayes_run_values=None,
    quote_exposure_values=None,
    as_cost: bool = False,
):
    q_net = deep_controller.q_net
    q_net.eval()

    def evaluate_state(mm_state: Dict[str, Any]) -> float:
        with torch.no_grad():
            s_tensor = deep_controller._state_to_tensor(mm_state)
            q_values = deep_controller._q_values_from_net(q_net, s_tensor)
            v = torch.max(q_values, dim=1).values.item()
        return -v if as_cost else v

    def set_feature(mm_state: Dict[str, Any], feature_name: str, value: float):
        if feature_name == "inventory":
            mm_state["inventory"] = float(value)
        elif feature_name == "spread":
            mm_state["spread"] = float(value)
        elif feature_name == "bid0":
            bid_sizes = list(mm_state["pure_mm_bid_sizes"])
            bid_sizes[0] = float(value)
            mm_state["pure_mm_bid_sizes"] = bid_sizes
        elif feature_name == "ask0":
            ask_sizes = list(mm_state["pure_mm_ask_sizes"])
            ask_sizes[0] = float(value)
            mm_state["pure_mm_ask_sizes"] = ask_sizes
        elif feature_name == "bidsize":
            mm_state["bidsize"] = float(value)
        elif feature_name == "asksize":
            mm_state["asksize"] = float(value)
        elif feature_name == "bayes_m_hat":
            mm_state["bayes_m_hat"] = float(value)
        elif feature_name == "bayes_expected_run_length":
            mm_state["bayes_expected_run_length"] = float(value)
        elif feature_name == "quote_exposure":
            mm_state["fill_imbalance_ewma"] = float(value)
        else:
            raise ValueError(f"Unknown feature: {feature_name}")

    def compute_slice(x_values, y_values, x_feature: str, y_feature: str):
        Z = np.zeros((len(y_values), len(x_values)), dtype=np.float32)
        for iy, y in enumerate(y_values):
            for ix, x in enumerate(x_values):
                mm_state = copy.deepcopy(base_state)
                set_feature(mm_state, x_feature, x)
                set_feature(mm_state, y_feature, y)
                Z[iy, ix] = evaluate_state(mm_state)
        return Z

    def plot_heatmap(x_values, y_values, Z, xlabel: str, ylabel: str, title: str):
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[
                float(np.min(x_values)),
                float(np.max(x_values)),
                float(np.min(y_values)),
                float(np.max(y_values)),
            ],
            cmap="viridis",
        )
        cbar_label = "Cost-to-go J(s)" if as_cost else "Value V(s)"
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=13)
        # ax.set_title(title)  # removed for paper export
        ax.grid(False)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "heatmap_" + _slug(title) + ".png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    def maybe_plot(x_values, y_values, x_feature, y_feature, xlabel, ylabel, title):
        if x_values is None or y_values is None:
            return
        if len(x_values) == 0 or len(y_values) == 0:
            return
        Z = compute_slice(x_values, y_values, x_feature, y_feature)
        plot_heatmap(x_values, y_values, Z, xlabel, ylabel, title)

    _plot_all_pairs(
        deep_controller, base_state, maybe_plot, as_cost,
        inv_values, spread_values, bid0_values, ask0_values,
        bid_size_values, ask_size_values,
        bayes_m_values, bayes_run_values, quote_exposure_values,
    )


def plot_value_surfaces_3d(
    deep_controller,
    base_state: Dict[str, Any],
    inv_values,
    spread_values,
    bid0_values=None,
    ask0_values=None,
    bid_size_values=None,
    ask_size_values=None,
    bayes_m_values=None,
    bayes_run_values=None,
    quote_exposure_values=None,
    as_cost: bool = False,
):
    q_net = deep_controller.q_net
    q_net.eval()

    def evaluate_state(mm_state: Dict[str, Any]) -> float:
        with torch.no_grad():
            s_tensor = deep_controller._state_to_tensor(mm_state)
            q_values = deep_controller._q_values_from_net(q_net, s_tensor)
            v = torch.max(q_values, dim=1).values.item()
        return -v if as_cost else v

    def set_feature(mm_state: Dict[str, Any], feature_name: str, value: float):
        if feature_name == "inventory":
            mm_state["inventory"] = float(value)
        elif feature_name == "spread":
            mm_state["spread"] = float(value)
        elif feature_name == "bid0":
            bid_sizes = list(mm_state["pure_mm_bid_sizes"])
            bid_sizes[0] = float(value)
            mm_state["pure_mm_bid_sizes"] = bid_sizes
        elif feature_name == "ask0":
            ask_sizes = list(mm_state["pure_mm_ask_sizes"])
            ask_sizes[0] = float(value)
            mm_state["pure_mm_ask_sizes"] = ask_sizes
        elif feature_name == "bidsize":
            mm_state["bidsize"] = float(value)
        elif feature_name == "asksize":
            mm_state["asksize"] = float(value)
        elif feature_name == "bayes_m_hat":
            mm_state["bayes_m_hat"] = float(value)
        elif feature_name == "bayes_expected_run_length":
            mm_state["bayes_expected_run_length"] = float(value)
        elif feature_name == "quote_exposure":
            mm_state["fill_imbalance_ewma"] = float(value)
        else:
            raise ValueError(f"Unknown feature: {feature_name}")

    def compute_surface(x_values, y_values, x_feature: str, y_feature: str):
        Z = np.zeros((len(y_values), len(x_values)), dtype=np.float32)
        for iy, y in enumerate(y_values):
            for ix, x in enumerate(x_values):
                mm_state = copy.deepcopy(base_state)
                set_feature(mm_state, x_feature, x)
                set_feature(mm_state, y_feature, y)
                Z[iy, ix] = evaluate_state(mm_state)
        X, Y = np.meshgrid(x_values, y_values)
        return X, Y, Z

    def plot_surface(x_values, y_values, Z, xlabel: str, ylabel: str, title: str):
        X, Y = np.meshgrid(x_values, y_values)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            X, Y, Z, cmap="viridis", edgecolor="none",
            alpha=0.92, linewidth=0, antialiased=True,
        )
        cbar_label = "Cost-to-go J(s)" if as_cost else "Value V(s)"
        fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, label=cbar_label)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_zlabel(cbar_label, fontsize=16)
        ax.tick_params(labelsize=13)
        # ax.set_title(title)  # removed for paper export
        ax.view_init(elev=28, azim=230)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "surface3d_" + _slug(title) + ".png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    def maybe_plot(x_values, y_values, x_feature, y_feature, xlabel, ylabel, title):
        if x_values is None or y_values is None:
            return
        if len(x_values) == 0 or len(y_values) == 0:
            return
        _, _, Z = compute_surface(x_values, y_values, x_feature, y_feature)
        plot_surface(x_values, y_values, Z, xlabel, ylabel, title)

    _plot_all_pairs(
        deep_controller, base_state, maybe_plot, as_cost,
        inv_values, spread_values, bid0_values, ask0_values,
        bid_size_values, ask_size_values,
        bayes_m_values, bayes_run_values, quote_exposure_values,
    )


def _plot_all_pairs(deep_controller, base_state, maybe_plot, as_cost,
                    inv_values, spread_values, bid0_values, ask0_values,
                    bid_size_values, ask_size_values,
                    bayes_m_values, bayes_run_values, quote_exposure_values):
    """Mesma sequencia de pares (x,y) das duas funcoes do Runner (fatorada p/ nao duplicar)."""
    is_pure_mm_state = (
        deep_controller.pure_mm
        and "pure_mm_bid_sizes" in base_state
        and "pure_mm_ask_sizes" in base_state
    )
    value_label = "cost" if as_cost else "value"

    if is_pure_mm_state:
        maybe_plot(spread_values, inv_values, "spread", "inventory",
                   "Spread", "Inventory", f"MM {value_label}: inventory x spread")
        maybe_plot(bid0_values, inv_values, "bid0", "inventory",
                   "Best-bid queue size", "Inventory", f"MM {value_label}: inventory x best-bid queue")
        maybe_plot(ask0_values, inv_values, "ask0", "inventory",
                   "Best-ask queue size", "Inventory", f"MM {value_label}: inventory x best-ask queue")
        maybe_plot(ask0_values, bid0_values, "ask0", "bid0",
                   "Best-ask queue size", "Best-bid queue size", f"MM {value_label}: best-bid x best-ask queue")
    else:
        maybe_plot(spread_values, inv_values, "spread", "inventory",
                   "Spread", "Inventory", f"MM {value_label}: inventory x spread")
        maybe_plot(bid_size_values, inv_values, "bidsize", "inventory",
                   "Bid size", "Inventory", f"MM {value_label}: inventory x bid size")
        maybe_plot(ask_size_values, inv_values, "asksize", "inventory",
                   "Ask size", "Inventory", f"MM {value_label}: inventory x ask size")
        maybe_plot(ask_size_values, bid_size_values, "asksize", "bidsize",
                   "Ask size", "Bid size", f"MM {value_label}: bid size x ask size")

    has_bayes_m = (getattr(deep_controller, "use_bayes_flow_signal", False)
                   and "bayes_m_hat" in base_state)
    has_bayes_run = (getattr(deep_controller, "use_bayes_flow_signal", False)
                     and "bayes_expected_run_length" in base_state)
    has_quote_exposure = (getattr(deep_controller, "use_fill_imbalance", False)
                          and "fill_imbalance_ewma" in base_state)

    if has_bayes_m:
        maybe_plot(bayes_m_values, inv_values, "bayes_m_hat", "inventory",
                   r"Bayes directional bias $\widehat{\iota}$", "Inventory",
                   f"Regime {value_label}: inventory x Bayes m_hat")
    if has_bayes_run:
        maybe_plot(bayes_run_values, inv_values, "bayes_expected_run_length", "inventory",
                   "Bayes expected run-length feature", "Inventory",
                   f"Regime {value_label}: inventory x Bayes run length")
    if has_quote_exposure:
        maybe_plot(quote_exposure_values, inv_values, "quote_exposure", "inventory",
                   "Quote-exposure imbalance", "Inventory",
                   f"Exposure {value_label}: inventory x quote exposure")
    if has_bayes_m and has_quote_exposure:
        maybe_plot(quote_exposure_values, bayes_m_values, "quote_exposure", "bayes_m_hat",
                   "Quote-exposure imbalance", r"Bayes directional bias $\widehat{\iota}$",
                   f"Regime/exposure {value_label}: Bayes m_hat x quote exposure")
    if has_bayes_m:
        maybe_plot(spread_values, bayes_m_values, "spread", "bayes_m_hat",
                   "Spread", r"Bayes directional bias $\widehat{\iota}$",
                   f"Regime/spread {value_label}: Bayes m_hat x spread")   # <- o PNG original


# =====================================================================
# CARREGAMENTO DO CONTROLLER
# =====================================================================
def load_controller():
    """Primario: o loader testado do projeto. Fallback: construcao manual resolvida."""
    try:
        from MM_GLFT_naive_comparison import make_controller_from_checkpoint
        ctrl = make_controller_from_checkpoint(
            CKPT,
            log_dir="runs_eval/viz_mix_off",
            use_time_update=True,
            min_time_interval=1.0,
            inv_limit_override=INV_LIMIT,
        )
        print("[OK] controller via make_controller_from_checkpoint")
        return ctrl
    except Exception as e:
        print(f"[WARN] make_controller_from_checkpoint falhou ({e!r}); "
              f"usando construcao manual.")
        return _build_controller_manual()


def _build_controller_manual():
    """Construcao 1:1 com os best_params do meta (fully-noisy dueling C51, input_dim=9)."""
    from dqn_distributional_with_throttle import DeepRLController
    ctrl = DeepRLController(
        level_offset=0,
        n_actions=6,
        gamma=0.999, lr=0.0015, weight_decay=0.01,
        epsilon_start=0.05, epsilon_min=0.05, epsilon_decay=0.995,
        batch_size=64, replay_capacity=100000, target_update_steps=2000,
        use_sarsa=False, use_double=True, use_dueling=True,
        use_prioritized_experience=True,
        use_noisy_net=True, use_factored_noise=True, use_fully_noisy=True,
        use_distributional=True, v_min=-3.0, v_max=3.0, atoms=101,
        n_neurons=256, n_hidden=1, activation="relu", elu_alpha=None, dropout_level=0.0,
        per_alpha_start=0.6, per_alpha_end=0.4, per_alpha_last_episode=2000,
        per_beta_start=0.4, per_beta_end=1.0, per_beta_last_episode=2000,
        n_steps=3, device=torch.device("cpu"), log_dir="runs/viz_tmp",
        use_tob_update=False, n_tob_moves=10, use_event_update=False, n_events=100,
        use_time_update=True, min_time_interval=1.0, use_mdp=True,
        use_flow_signal=False, use_fast_flow_signal=False,
        use_bayes_flow_signal=True,
        bayes_flow_feature_keys=["bayes_m_hat", "bayes_expected_run_length"],
        use_fill_imbalance=True, use_g_clip=True, use_per_priority_clip=True,
        pure_mm=True, inv_limit=INV_LIMIT,
        pure_mm_offsets=[(-1, -1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)],
        enable_learning=False,
    )
    # RISCO: se o meta tivesse objetos nao-builtin, trocar weights_only=False.
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    ctrl.q_net.load_state_dict(ckpt["q_net"], strict=True)
    ctrl.target_net.load_state_dict(ckpt["target_net"], strict=True)
    ctrl.epsilon = 0.0
    ctrl.q_net.eval()
    ctrl.target_net.eval()
    print("[OK] controller via construcao manual")
    return ctrl


# =====================================================================
# (A) gera os plots de valor
# =====================================================================
def make_value_plots(ctrl, base_state=None, queue_values=None, queue_mode="auto",
                     queue_min=None, queue_max=None, queue_points=35):
    base_state = copy.deepcopy(base_state if base_state is not None else BASE_STATE_NOMINAL)
    # garante as chaves minimas exigidas pelo _state_to_tensor (pure_mm)
    base_state.setdefault("spread", 2.0)
    base_state.setdefault("inventory", 0.0)
    base_state.setdefault("pure_mm_bid_sizes", [5.0, 5.0])
    base_state.setdefault("pure_mm_ask_sizes", [5.0, 5.0])
    base_state.setdefault("bayes_m_hat", 0.0)
    base_state.setdefault("bayes_expected_run_length", 0.7)
    base_state.setdefault("fill_imbalance_ewma", 0.0)
    print(f"[A] base_state: spread={base_state['spread']:.2f} inv={base_state['inventory']:.2f} "
          f"bid0={base_state['pure_mm_bid_sizes'][0]:.2f} ask0={base_state['pure_mm_ask_sizes'][0]:.2f} "
          f"m_hat={base_state['bayes_m_hat']:.3f} run={base_state['bayes_expected_run_length']:.3f} "
          f"qexp={base_state['fill_imbalance_ewma']:.3f}")
    inv_limit = int(getattr(ctrl, "inv_limit", INV_LIMIT) or INV_LIMIT)
    inv_values = np.arange(-inv_limit, inv_limit + 1, 1)

    base_spread = float(base_state.get("spread", 2.0))
    spread_values = np.arange(max(1.0, base_spread - 3.0), base_spread + 3.0 + 1.0, 1.0)

    base0 = max(base_state["pure_mm_bid_sizes"][0], base_state["pure_mm_ask_sizes"][0], 1.0)
    queue_points = max(2, int(queue_points))
    queue_mode = str(queue_mode or "auto").lower()
    if queue_min is not None or queue_max is not None:
        q_min = 1.0 if queue_min is None else float(queue_min)
        q_max = 4.0 if queue_max is None else float(queue_max)
        if q_max <= q_min:
            raise ValueError(f"queue_max precisa ser > queue_min; recebi {q_min}..{q_max}")
        offset_values = np.linspace(q_min, q_max, queue_points)
        queue_source = f"fixed {q_min:g}..{q_max:g}"
    elif queue_mode == "runner" and queue_values is not None:
        offset_values = np.asarray(queue_values, dtype=float)
        queue_source = "runner percentiles 5%-95%"
    elif queue_mode == "auto" and queue_values is not None:
        offset_values = np.asarray(queue_values, dtype=float)
        queue_source = "auto runner percentiles 5%-95%"
    elif queue_mode in ("paper", "local", "old"):
        offset_values = np.linspace(1.0, 4.0, queue_points)
        queue_source = "paper/local fixed 1..4"
    elif queue_mode in ("runner", "auto"):
        offset_values = np.linspace(1.0, 4.0, queue_points)
        queue_source = "paper/local fixed 1..4 (runner grid unavailable)"
    elif queue_mode == "wide":
        offset_values = np.linspace(0.0, max(20.0, 2.0 * base0), queue_points)
        queue_source = "wide fixed 0..max(20,2*base0)"
    else:
        raise ValueError("queue_mode deve ser auto, runner, paper ou wide")
    print(f"[A] queue grid ({queue_source}): {offset_values[0]:.3g} -> "
          f"{offset_values[-1]:.3g} ({len(offset_values)} pts)")

    bayes_m_values = np.linspace(-1.0, 1.0, 41)
    bayes_run_values = np.linspace(0.0, 1.0, 41)
    quote_exposure_values = np.linspace(-1.0, 1.0, 41)

    kwargs = dict(
        deep_controller=ctrl, base_state=base_state,
        inv_values=inv_values, spread_values=spread_values,
        bid0_values=offset_values, ask0_values=offset_values,
        bayes_m_values=bayes_m_values, bayes_run_values=bayes_run_values,
        quote_exposure_values=quote_exposure_values, as_cost=False,
    )
    print("[A] heatmaps 2D...")
    plot_value_slices_2d(**kwargs)
    print("[A] superficies 3D...")
    plot_value_surfaces_3d(**kwargs)
    print(f"[A] plots de valor salvos em {OUT_DIR}")


# =====================================================================
# (B) gera os plots de inventario + PnL total (rollouts de avaliacao)
# =====================================================================
def make_inv_pnl_plots(ctrl):
    from MM_LOB_SIM import simulate_LOB_with_MM
    from CONFIG_MM import lam, mu, delta, GLOBAL_SEED, mean_size_LO
    try:
        from CONFIG_MM import USE_QRM, qrm_params
    except Exception:
        USE_QRM, qrm_params = False, {}

    # --- OPCIONAL: regime exponencial (treino foi com regime). Se quiser usar,
    #     descomente e ajuste; por padrao roda estacionario buy_mo_prob=BUY_MO_PROB.
    # from regime_switching_stress_test_2 import make_regime_schedule
    # def regime_bmp(seed): return make_regime_schedule(seed=seed+999999, n_mo_events=N_STEPS,
    #     L_min=10, alpha=1.5, p_lo=0.2, p_hi=0.8, distribution="exponential", exp_rate=1/60.0)

    inv_list, pnl_list, terminal = [], [], []
    for i in range(N_EVAL_EPISODES):
        seed = int(GLOBAL_SEED) + i
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        sim_kwargs = dict(
            lam=lam, mu=mu, delta=delta,
            number_tick_levels=50, n_priority_ranks=100, number_levels_to_store=20,
            p0=100, mean_size_LO=mean_size_LO,
            iterations=N_STEPS, iterations_to_equilibrium=N_STEPS_TO_EQUIL,
            path_save_files=None, label_simulation=None,
            controller=ctrl, random_seed=seed, buy_mo_prob=BUY_MO_PROB,
        )
        if USE_QRM:
            sim_kwargs["qrm_params"] = qrm_params
        _, _, mm_df = simulate_LOB_with_MM(**sim_kwargs)
        inv_list.append(mm_df["MM_Inventory"].to_numpy())
        pnl_list.append(mm_df["MM_TotalPnL"].to_numpy())
        terminal.append(float(mm_df["MM_TotalPnL"].iloc[-1]))
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[B] sim {i+1}/{N_EVAL_EPISODES}  PnL_final={terminal[-1]:+.4f}")

    T = min(len(a) for a in inv_list)
    inv_mat = np.stack([a[:T] for a in inv_list])
    pnl_mat = np.stack([a[:T] for a in pnl_list])
    t = np.arange(T)

    # --- inventario: banda media +/- 1 std (step) ---
    mean_i, std_i = inv_mat.mean(0), inv_mat.std(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.step(t, mean_i, where="post", color="crimson", label="mean inventory")
    ax.fill_between(t, mean_i - std_i, mean_i + std_i, step="post", alpha=0.15, color="crimson")
    ax.axhline(0, color="black", alpha=0.3)
    ax.axhline(INV_LIMIT, color="gray", ls="--", alpha=0.4)
    ax.axhline(-INV_LIMIT, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("Simulation step"); ax.set_ylabel("MM_Inventory")
    ax.set_title(f"Mean Inventory(t) +/- 1 std (N={N_EVAL_EPISODES})")
    ax.grid(True, ls="--", alpha=0.4); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "eval_inventory_trajectory.png"), dpi=150)
    plt.close(fig)

    # --- PnL total (cumulativo): banda media +/- 1 std (linha) ---
    mean_p, std_p = pnl_mat.mean(0), pnl_mat.std(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, mean_p, color="royalblue", label="mean TotalPnL")
    ax.fill_between(t, mean_p - std_p, mean_p + std_p, alpha=0.15, color="royalblue")
    ax.axhline(0, color="black", alpha=0.3)
    ax.set_xlabel("Simulation step"); ax.set_ylabel("MM_TotalPnL")
    ax.set_title(f"Mean TotalPnL(t) +/- 1 std (N={N_EVAL_EPISODES})")
    ax.grid(True, ls="--", alpha=0.4); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "eval_total_pnl_trajectory.png"), dpi=150)
    plt.close(fig)

    # --- distribuicao do PnL terminal ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(terminal, bins=min(40, max(5, N_EVAL_EPISODES)), alpha=0.55,
            color="seagreen", density=True, edgecolor="black", linewidth=0.3)
    ax.axvline(float(np.mean(terminal)), color="black", ls="--",
               label=f"mean={np.mean(terminal):+.3f}")
    ax.set_xlabel("Terminal PnL"); ax.set_ylabel("Density")
    ax.set_title(f"Terminal PnL distribution (N={N_EVAL_EPISODES})")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "eval_terminal_pnl_dist.png"), dpi=150)
    plt.close(fig)
    print(f"[B] plots de inventario/PnL salvos em {OUT_DIR}  "
          f"(mean terminal PnL = {np.mean(terminal):+.4f})")


def main():
    import argparse
    global CKPT, OUT_DIR, N_EVAL_EPISODES

    ap = argparse.ArgumentParser(
        description="Gera plots (valor + inv/PnL) de um checkpoint DQN, no padrao do Runner_REGIME.")
    ap.add_argument("ckpt", nargs="?", default="final",
                    help="'final' (default), 'best', ou caminho completo do .pt")
    ap.add_argument("--ckpt", "--checkpoint", dest="ckpt_option", default=None,
                    help="checkpoint explicito; sobrescreve o argumento posicional")
    ap.add_argument("--out", default=None,
                    help="pasta de saida (default: derivada do nome do ckpt)")
    ap.add_argument("--episodes", type=int, default=N_EVAL_EPISODES,
                    help=f"rollouts de eval p/ inv/PnL (default: {N_EVAL_EPISODES})")
    ap.add_argument("--no-sim", action="store_true",
                    help="pula a parte (B) inv/PnL; gera so os plots de valor")
    ap.add_argument("--base", choices=["nominal", "rollout"], default="nominal",
                    help="origem do base_state das fatias (default: nominal = estado neutro fixo; "
                         "rollout = replica o Runner usando um episodio greedy)")
    ap.add_argument("--queue-mode", choices=["auto", "runner", "paper", "wide"], default="paper",
                    help="grade das filas L1: paper usa 1..4 (default, comparavel ao plot antigo); "
                         "auto usa runner se houver rollout, senao 1..4; "
                         "runner usa percentis 5%%-95%% do episodio; paper usa 1..4; "
                         "wide usa 0..max(20,2*base0)")
    ap.add_argument("--queue-min", type=float, default=None,
                    help="limite inferior fixo da grade de filas L1; sobrescreve --queue-mode")
    ap.add_argument("--queue-max", type=float, default=None,
                    help="limite superior fixo da grade de filas L1; sobrescreve --queue-mode")
    ap.add_argument("--queue-points", type=int, default=35,
                    help="numero de pontos na grade de filas L1 (default: 35)")
    ap.add_argument("--seed", type=int, default=LAST_EP_SEED,
                    help=f"seed do rollout de base_state (default {LAST_EP_SEED} = "
                         f"ultimo ep de treino do final.pt)")
    args = ap.parse_args()

    ckpt_arg = args.ckpt_option or args.ckpt
    CKPT = KNOWN_CKPTS.get(ckpt_arg, ckpt_arg)       # atalho ('final'/'best') ou caminho
    OUT_DIR = args.out or (_out_dir_for(CKPT) + "_" + _queue_out_tag(args))
    N_EVAL_EPISODES = int(args.episodes)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isfile(CKPT):
        print(f"[ERRO] checkpoint nao encontrado: {CKPT}")
        sys.exit(1)

    print(f"Checkpoint: {CKPT}")
    print(f"Saida:      {OUT_DIR}\n")
    ctrl = load_controller()

    base_state = None
    queue_values = None
    if args.base == "rollout":
        try:
            base_state, queue_values = derive_base_state_from_rollout(ctrl, seed=args.seed)
            print("[A] base_state derivado de rollout (replica metodologia do Runner)")
            if queue_values is not None:
                print(f"[A] queue grid derivada do rollout: {queue_values[0]:.3g} -> "
                      f"{queue_values[-1]:.3g} ({len(queue_values)} pts)")
        except Exception as e:
            print(f"[A] rollout p/ base_state falhou ({e!r}); usando BASE_STATE_NOMINAL")
            base_state = None
            queue_values = None

    try:
        make_value_plots(
            ctrl,
            base_state,
            queue_values=queue_values,
            queue_mode=args.queue_mode,
            queue_min=args.queue_min,
            queue_max=args.queue_max,
            queue_points=args.queue_points,
        )
    except Exception as e:
        import traceback
        print(f"[A] FALHOU nos plots de valor: {e!r}")
        traceback.print_exc()

    if not args.no_sim:
        try:
            make_inv_pnl_plots(ctrl)
        except Exception as e:
            import traceback
            print(f"[B] FALHOU nos plots de inv/PnL (o env/simulador pode precisar de "
                  f"ajuste): {e!r}")
            traceback.print_exc()
    else:
        print("[B] pulado (--no-sim)")

    print("\nPronto.")


if __name__ == "__main__":
    main()
