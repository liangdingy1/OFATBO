import itertools
import json
import math
import os
import random
import warnings
from argparse import Namespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=UserWarning, module="dragonfly.utils.oper_utils")

try:
    from dragonfly import load_config
    from dragonfly.exd.experiment_caller import CPFunctionCaller
    from dragonfly.opt.gp_bandit import CPGPBandit
    DRAGONFLY_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime guard for environments without dragonfly
    DRAGONFLY_AVAILABLE = False
    load_config = None
    CPFunctionCaller = None
    CPGPBandit = None


BATCH_SIZE = 5
DEFAULT_EI_XI = 0.01
DEFAULT_UCB_KAPPA = 2.576
DEFAULT_THRESHOLDS = (0.95, 0.97, 0.99)


# -----------------------------------------------------------------------------
# Acquisition utilities
# -----------------------------------------------------------------------------
def ucb(mu: float, sigma: float, kappa: float = DEFAULT_UCB_KAPPA) -> float:
    return float(mu + kappa * sigma)


def ei(mu: float, sigma: float, tau: float, xi: float = DEFAULT_EI_XI) -> float:
    sigma = float(sigma)
    if sigma <= 1e-12:
        return 0.0
    z = (mu - tau - xi) / sigma
    val = (mu - tau - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return float(max(val, 0.0))


# -----------------------------------------------------------------------------
# Benchmark-function helpers
# -----------------------------------------------------------------------------
def _rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_m @ ry_m @ rx_m


_ROT = _rotation_matrix_xyz(25.0, -35.0, 40.0)


def _hartmann3_raw(x: np.ndarray) -> float:
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    a = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ])
    p = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828],
    ])
    return float(np.sum(alpha * np.exp(-np.sum(a * (x - p) ** 2, axis=1))))


def _ackley3_raw(x: np.ndarray) -> float:
    a = 20.0
    b = 0.2
    c = 2.0 * math.pi
    d = 3
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    ackley = term1 + term2 + a + math.e
    return float(-ackley)


def _branin_weak_z_raw(x: np.ndarray) -> float:
    # x[0] in [-5, 10], x[1] in [0, 15], x[2] in [0, 1]
    x1, x2, x3 = x
    a = 1.0
    b = 5.1 / (4.0 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    branin = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
    weak_z = 0.8 * math.exp(-((x3 - 0.68) ** 2) / 0.03)
    return float(-branin + weak_z)


def _rotated_ridge_raw(x: np.ndarray) -> float:
    z = _ROT @ x
    center = np.array([0.35, -0.15, 0.10])
    delta = z - center
    raw = -((delta[0] ** 2) / 0.004 + (delta[1] ** 2) / 0.18 + (delta[2] ** 2) / 0.55)
    return float(raw)


def _rotated_rosenbrock_raw(x: np.ndarray) -> float:
    z = _ROT @ x + np.array([0.9, 0.9, 0.9])
    raw = -(
        100.0 * (z[1] - z[0] ** 2) ** 2 + (1.0 - z[0]) ** 2
        + 40.0 * (z[2] - z[1] ** 2) ** 2 + (1.0 - z[1]) ** 2
    )
    return float(raw)


def _hetero_cardinality_stress_raw(x: np.ndarray) -> float:
    # Designed so that x1/x2 matter sharply, while x3 is broad and low-cardinality.
    x1, x2, x3 = x
    peak1 = 1.7 * math.exp(-((x1 - 0.72) ** 2) / 0.004 - ((x2 - 0.28) ** 2) / 0.006)
    peak2 = 0.95 * math.exp(-((x1 - 0.18) ** 2) / 0.012 - ((x2 - 0.77) ** 2) / 0.012)
    broad_z = 0.18 * math.exp(-((x3 - 0.70) ** 2) / 0.20)
    coupling = 0.10 * math.sin(4.0 * math.pi * x1) * math.cos(2.0 * math.pi * x2)
    return float(peak1 + peak2 + broad_z + coupling)


def _linspace_list(start: float, stop: float, n: int) -> List[float]:
    return [float(x) for x in np.linspace(start, stop, n)]


def _scale_to_0_100(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-12:
        return np.zeros_like(values) + 50.0
    return (values - vmin) / (vmax - vmin) * 100.0


def _build_candidate_table(dim_names: Sequence[str], levels: Sequence[Sequence[float]]) -> pd.DataFrame:
    idx_names = [f"{d}_idx" for d in dim_names]
    rows = []
    point_id = 0
    for idx_tuple in itertools.product(*[range(len(v)) for v in levels]):
        row = {"point_id": point_id}
        for d_name, idx_name, vals, idx in zip(dim_names, idx_names, levels, idx_tuple):
            row[d_name] = vals[idx]
            row[idx_name] = int(idx)
        rows.append(row)
        point_id += 1
    return pd.DataFrame(rows)


def _attach_true_values(candidate_table: pd.DataFrame, dim_names: Sequence[str], raw_func) -> pd.DataFrame:
    raw_vals = []
    for row in candidate_table.itertuples(index=False):
        x = np.array([getattr(row, d) for d in dim_names], dtype=float)
        raw_vals.append(raw_func(x))
    candidate_table = candidate_table.copy()
    candidate_table["y_raw"] = np.array(raw_vals, dtype=float)
    candidate_table["y_true"] = _scale_to_0_100(candidate_table["y_raw"].values.astype(float))
    return candidate_table


def build_function_registry() -> Dict[str, Dict]:
    registry = {}

    common_012_levels = [
        _linspace_list(0.0, 1.0, 25),
        _linspace_list(0.0, 1.0, 20),
        _linspace_list(0.0, 1.0, 5),
    ]

    canonical_specs = {
        "hartmann3": {
            "family": "canonical",
            "dim_names": ["x1", "x2", "x3"],
            "levels": common_012_levels,
            "raw_func": _hartmann3_raw,
        },
        "hetero_cardinality_stress": {
            "family": "hetero_cardinality_stress",
            "dim_names": ["x1", "x2", "x3"],
            "levels": common_012_levels,
            "raw_func": _hetero_cardinality_stress_raw,
        },
    }

    ackley_levels = [
        _linspace_list(-5.0, 5.0, 25),
        _linspace_list(-5.0, 5.0, 20),
        _linspace_list(-5.0, 5.0, 5),
    ]
    canonical_specs["ackley3"] = {
        "family": "canonical",
        "dim_names": ["x1", "x2", "x3"],
        "levels": ackley_levels,
        "raw_func": _ackley3_raw,
    }

    branin_levels = [
        _linspace_list(-5.0, 10.0, 25),
        _linspace_list(0.0, 15.0, 20),
        _linspace_list(0.0, 1.0, 5),
    ]
    canonical_specs["branin_weak_z"] = {
        "family": "canonical",
        "dim_names": ["x1", "x2", "x3"],
        "levels": branin_levels,
        "raw_func": _branin_weak_z_raw,
    }

    rot_levels = [
        _linspace_list(-1.5, 1.5, 25),
        _linspace_list(-1.5, 1.5, 20),
        _linspace_list(-1.5, 1.5, 5),
    ]
    canonical_specs["rotated_ridge"] = {
        "family": "interaction_stress",
        "dim_names": ["x1", "x2", "x3"],
        "levels": rot_levels,
        "raw_func": _rotated_ridge_raw,
    }
    canonical_specs["rotated_rosenbrock"] = {
        "family": "interaction_stress",
        "dim_names": ["x1", "x2", "x3"],
        "levels": rot_levels,
        "raw_func": _rotated_rosenbrock_raw,
    }

    for name, spec in canonical_specs.items():
        candidate_table = _build_candidate_table(spec["dim_names"], spec["levels"])
        candidate_table = _attach_true_values(candidate_table, spec["dim_names"], spec["raw_func"])
        domain_vars = [
            {"name": d, "type": "discrete_numeric", "items": levels}
            for d, levels in zip(spec["dim_names"], spec["levels"])
        ]
        coord_to_pid = {}
        pid_to_coord = {}
        for row in candidate_table.itertuples(index=False):
            coord = tuple(getattr(row, d) for d in spec["dim_names"])
            coord_to_pid[coord] = int(row.point_id)
            pid_to_coord[int(row.point_id)] = coord
        y_max = float(candidate_table["y_true"].max())
        best_ids = candidate_table.loc[candidate_table["y_true"] == y_max, "point_id"].astype(int).tolist()
        registry[name] = {
            "name": name,
            "family": spec["family"],
            "dim_names": spec["dim_names"],
            "idx_names": [f"{d}_idx" for d in spec["dim_names"]],
            "levels": spec["levels"],
            "domain_vars": domain_vars,
            "candidate_table": candidate_table,
            "coord_to_pid": coord_to_pid,
            "pid_to_coord": pid_to_coord,
            "global_max_value": y_max,
            "global_max_point_ids": best_ids,
            "response_range": float(candidate_table["y_true"].max() - candidate_table["y_true"].min()),
        }
    return registry


# -----------------------------------------------------------------------------
# Noise utilities
# -----------------------------------------------------------------------------
def sample_observation(y_true: float, point_row: pd.Series, spec: Dict, noise_cfg: Optional[Dict], rng: np.random.Generator) -> float:
    if noise_cfg is None or noise_cfg.get("mode", "none") == "none":
        return float(y_true)

    mode = noise_cfg.get("mode", "none")
    level = float(noise_cfg.get("level", 0.0))
    base_sigma = level * spec["response_range"]

    if mode == "additive_gaussian":
        sigma = base_sigma
    elif mode == "heteroscedastic_gaussian":
        coords = np.array([
            point_row[idx_name] / max(1, len(levels) - 1)
            for idx_name, levels in zip(spec["idx_names"], spec["levels"])
        ])
        radial = float(np.mean(np.abs(coords - 0.5)))
        sigma = base_sigma * (0.75 + 1.5 * radial)
    else:
        raise ValueError(f"Unknown noise mode: {mode}")

    noisy = float(y_true + rng.normal(0.0, sigma))
    return float(np.clip(noisy, 0.0, 100.0))


# -----------------------------------------------------------------------------
# Dragonfly helpers
# -----------------------------------------------------------------------------
def _require_dragonfly():
    if not DRAGONFLY_AVAILABLE:
        raise ImportError(
            "dragonfly is not installed in the current Python environment. "
            "Please run this code in your local environment where dragonfly is available."
        )


def _canonicalize_raw_point(x: Sequence[float], spec: Dict) -> Tuple[float, ...]:
    coord = []
    for val, levels in zip(x, spec["levels"]):
        nearest = min(levels, key=lambda v: abs(float(v) - float(val)))
        coord.append(float(nearest))
    return tuple(coord)


def build_dragonfly_posterior(spec: Dict, observed_df: pd.DataFrame, hp_tune: str = "ml"):
    _require_dragonfly()
    config = load_config({"domain": spec["domain_vars"]})
    options = Namespace(gpb_hp_tune_criterion=hp_tune)
    func_caller = CPFunctionCaller(None, config.domain, domain_orderings=config.domain_orderings)
    opt = CPGPBandit(func_caller, "default", ask_tell_mode=True, options=options)
    opt.worker_manager = None
    opt._set_up()
    opt.initialise()

    for row in observed_df.itertuples(index=False):
        x = [getattr(row, d) for d in spec["dim_names"]]
        opt.tell([(x, float(row.y_obs))])
        opt.step_idx += 1

    opt._build_new_model()
    opt._set_next_gp()

    results = []
    tau = float(observed_df["y_obs"].max())
    for row in spec["candidate_table"].itertuples(index=False):
        x_raw = [getattr(row, d) for d in spec["dim_names"]]
        x_input = [opt.func_caller.get_processed_domain_point_from_raw(x_raw)]
        mu, stdev = opt.gp.eval(x_input, uncert_form="std")
        mu = float(mu[0])
        stdev = float(stdev[0])
        results.append({
            "point_id": int(row.point_id),
            "mu": mu,
            "stdev": stdev,
            "value_ei": ei(mu, stdev, tau),
            "value_ucb": ucb(mu, stdev),
        })

    acq_df = spec["candidate_table"].merge(pd.DataFrame(results), on="point_id", how="left")
    return opt, acq_df


def build_dragonfly_seq_optimizer(spec: Dict, observed_df: pd.DataFrame, hp_tune: str = "ml"):
    _require_dragonfly()
    config = load_config({"domain": spec["domain_vars"]})
    options = Namespace(gpb_hp_tune_criterion=hp_tune)
    func_caller = CPFunctionCaller(None, config.domain, domain_orderings=config.domain_orderings)
    opt = CPGPBandit(func_caller, "default", ask_tell_mode=True, options=options)
    opt.worker_manager = None
    opt._set_up()
    opt.initialise()
    for row in observed_df.itertuples(index=False):
        x = [getattr(row, d) for d in spec["dim_names"]]
        opt.tell([(x, float(row.y_obs))])
        opt.step_idx += 1
    return opt


# -----------------------------------------------------------------------------
# OFAT helpers
# -----------------------------------------------------------------------------
def _unsampled_acquisition_df(acq_df: pd.DataFrame, observed_ids: set) -> pd.DataFrame:
    return acq_df.loc[~acq_df["point_id"].isin(observed_ids)].copy().reset_index(drop=True)


def _compute_coarse_grid(acq_values: np.ndarray, shape: Tuple[int, ...]):
    arr = acq_values.reshape(shape)
    block_sizes = [int(math.ceil(n / BATCH_SIZE)) for n in shape]
    pad_widths = []
    padded = arr
    for axis, (n, block) in enumerate(zip(shape, block_sizes)):
        target = BATCH_SIZE * block
        total_pad = target - n
        pre = total_pad // 2
        post = total_pad - pre
        pad_widths.append((pre, post))
    padded = np.pad(arr, pad_widths, mode="edge")

    coarse = np.zeros(tuple([BATCH_SIZE] * len(shape)), dtype=float)
    for coarse_idx in itertools.product(*[range(BATCH_SIZE) for _ in shape]):
        slices = tuple(slice(i * block, (i + 1) * block) for i, block in zip(coarse_idx, block_sizes))
        coarse[coarse_idx] = float(np.mean(padded[slices]))
    return coarse, block_sizes, pad_widths


def _anchor_coarse_index(anchor_row: pd.Series, spec: Dict, block_sizes: Sequence[int], pad_widths: Sequence[Tuple[int, int]]) -> Tuple[int, ...]:
    coarse_idx = []
    for idx_name, block, pad in zip(spec["idx_names"], block_sizes, pad_widths):
        idx_val = int(anchor_row[idx_name])
        coarse_val = int((idx_val + pad[0]) // block)
        coarse_val = max(0, min(BATCH_SIZE - 1, coarse_val))
        coarse_idx.append(coarse_val)
    return tuple(coarse_idx)


def _direction_scores(acq_df: pd.DataFrame, anchor_row: pd.Series, spec: Dict, acq_field: str, use_gridding: bool) -> List[Tuple[str, float]]:
    dim_names = spec["dim_names"]
    if use_gridding:
        shape = tuple(len(levels) for levels in spec["levels"])
        coarse, block_sizes, pad_widths = _compute_coarse_grid(acq_df[acq_field].values.astype(float), shape)
        coarse_idx = _anchor_coarse_index(anchor_row, spec, block_sizes, pad_widths)
        scores = []
        for axis, dim_name in enumerate(dim_names):
            score = float(np.take(coarse, indices=coarse_idx[axis], axis=axis).mean())
            scores.append((dim_name, score))
        return sorted(scores, key=lambda x: -x[1])

    scores = []
    for dim_name in dim_names:
        mask = np.ones(len(acq_df), dtype=bool)
        for other_dim in dim_names:
            if other_dim == dim_name:
                continue
            mask &= acq_df[other_dim].values == anchor_row[other_dim]
        line_df = acq_df.loc[mask]
        scores.append((dim_name, float(line_df[acq_field].mean())))
    return sorted(scores, key=lambda x: -x[1])


def _line_dataframe(acq_df: pd.DataFrame, anchor_row: pd.Series, spec: Dict, varying_dim: str) -> pd.DataFrame:
    mask = np.ones(len(acq_df), dtype=bool)
    for dim_name in spec["dim_names"]:
        if dim_name == varying_dim:
            continue
        mask &= acq_df[dim_name].values == anchor_row[dim_name]
    idx_name = f"{varying_dim}_idx"
    return acq_df.loc[mask].sort_values(idx_name).reset_index(drop=True)


def _select_points_from_line(line_df: pd.DataFrame, anchor_row: pd.Series, observed_ids: set, acq_field: str, use_nonadjacent: bool) -> pd.DataFrame:
    varying_dim = None
    for col in line_df.columns:
        if col.endswith("_idx"):
            base = col[:-4]
            if line_df[base].nunique() > 1:
                varying_dim = base
                break
    if varying_dim is None:
        raise ValueError("Could not infer varying dimension for OFAT line selection.")
    varying_idx_name = f"{varying_dim}_idx"

    available = line_df.loc[~line_df["point_id"].isin(observed_ids)].copy()
    available = available.loc[available["point_id"] != int(anchor_row["point_id"])].copy()

    selected_rows = [anchor_row]
    while len(selected_rows) < BATCH_SIZE and not available.empty:
        best_pos = available[acq_field].astype(float).idxmax()
        best_row = available.loc[best_pos]
        selected_rows.append(best_row)

        if use_nonadjacent:
            chosen_idx = int(best_row[varying_idx_name])
            drop_mask = available[varying_idx_name].isin([chosen_idx - 1, chosen_idx, chosen_idx + 1])
            candidate_remaining = available.loc[~drop_mask].copy()
            needed_after = BATCH_SIZE - len(selected_rows)
            if len(candidate_remaining) >= needed_after:
                available = candidate_remaining
            else:
                available = available.loc[available["point_id"] != int(best_row["point_id"])].copy()
        else:
            available = available.loc[available["point_id"] != int(best_row["point_id"])].copy()

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def propose_ofat_batch(
    spec: Dict,
    observed_df: pd.DataFrame,
    anchor_mode: str = "ei",
    direction_mode: str = "ucb",
    use_gridding: bool = True,
    use_nonadjacent: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    _, acq_df = build_dragonfly_posterior(spec, observed_df)
    observed_ids = set(observed_df["point_id"].astype(int).tolist())
    unknown_df = _unsampled_acquisition_df(acq_df, observed_ids)
    if len(unknown_df) == 0:
        raise ValueError("No unsampled points remain.")

    anchor_field = "value_ei" if anchor_mode == "ei" else "value_ucb"
    direction_field = "value_ei" if direction_mode == "ei" else "value_ucb"

    anchor_pos = unknown_df[anchor_field].astype(float).idxmax()
    anchor_row = unknown_df.loc[anchor_pos]

    ranked_dirs = _direction_scores(acq_df, anchor_row, spec, direction_field, use_gridding)

    chosen_dim = None
    chosen_line = None
    for dim_name, _ in ranked_dirs:
        line_df = _line_dataframe(acq_df, anchor_row, spec, dim_name)
        filtered = line_df.loc[~line_df["point_id"].isin(observed_ids)].copy()
        filtered = filtered.loc[filtered["point_id"] != int(anchor_row["point_id"])].copy()
        if len(filtered) >= BATCH_SIZE - 1:
            chosen_dim = dim_name
            chosen_line = line_df
            break

    if chosen_line is None:
        # robust fallback: use best available line, then fill remaining with global acquisition ranking
        dim_name = ranked_dirs[0][0]
        chosen_dim = dim_name
        chosen_line = _line_dataframe(acq_df, anchor_row, spec, dim_name)

    selected = _select_points_from_line(chosen_line, anchor_row, observed_ids, direction_field, use_nonadjacent)
    if len(selected) < BATCH_SIZE:
        needed = BATCH_SIZE - len(selected)
        used_ids = set(selected["point_id"].astype(int).tolist()) | observed_ids
        fill_df = unknown_df.loc[~unknown_df["point_id"].isin(used_ids)].copy()
        fill_df = fill_df.sort_values(direction_field, ascending=False).head(needed)
        if not fill_df.empty:
            selected = pd.concat([selected, fill_df], ignore_index=True)

    selected = selected.head(BATCH_SIZE).copy().reset_index(drop=True)
    meta = {
        "anchor_point_id": int(anchor_row["point_id"]),
        "selected_dim": chosen_dim,
        "anchor_acq": anchor_mode,
        "direction_acq": direction_mode,
        "used_gridding": bool(use_gridding),
        "used_nonadjacent": bool(use_nonadjacent),
    }
    return selected, meta


def propose_topk_batch(spec: Dict, observed_df: pd.DataFrame, acq_mode: str = "ei", top_k: int = BATCH_SIZE) -> Tuple[pd.DataFrame, Dict]:
    _, acq_df = build_dragonfly_posterior(spec, observed_df)
    observed_ids = set(observed_df["point_id"].astype(int).tolist())
    unknown_df = _unsampled_acquisition_df(acq_df, observed_ids)
    acq_field = "value_ei" if acq_mode == "ei" else "value_ucb"
    selected = unknown_df.sort_values(acq_field, ascending=False).head(top_k).copy().reset_index(drop=True)
    meta = {
        "anchor_point_id": None,
        "selected_dim": None,
        "anchor_acq": acq_mode,
        "direction_acq": acq_mode,
        "used_gridding": False,
        "used_nonadjacent": False,
    }
    return selected, meta


# -----------------------------------------------------------------------------
# Method registry
# -----------------------------------------------------------------------------
def get_method_registry() -> Dict[str, Dict]:
    return {
        "seq_dragonfly": {
            "type": "sequential_dragonfly",
            "batch_size": 1,
        },
        "batch_ei_top5": {
            "type": "topk_batch",
            "acq_mode": "ei",
            "batch_size": BATCH_SIZE,
        },
        "batch_ucb_top5": {
            "type": "topk_batch",
            "acq_mode": "ucb",
            "batch_size": BATCH_SIZE,
        },
        "ofat_full": {
            "type": "ofat",
            "anchor_mode": "ei",
            "direction_mode": "ucb",
            "use_gridding": True,
            "use_nonadjacent": True,
            "batch_size": BATCH_SIZE,
        },
        "ofat_no_gridding": {
            "type": "ofat",
            "anchor_mode": "ei",
            "direction_mode": "ucb",
            "use_gridding": False,
            "use_nonadjacent": True,
            "batch_size": BATCH_SIZE,
        },
        "ofat_no_nonadjacent": {
            "type": "ofat",
            "anchor_mode": "ei",
            "direction_mode": "ucb",
            "use_gridding": True,
            "use_nonadjacent": False,
            "batch_size": BATCH_SIZE,
        },
        "ofat_ucb_ucb": {
            "type": "ofat",
            "anchor_mode": "ucb",
            "direction_mode": "ucb",
            "use_gridding": True,
            "use_nonadjacent": True,
            "batch_size": BATCH_SIZE,
        },
        "ofat_ei_ei": {
            "type": "ofat",
            "anchor_mode": "ei",
            "direction_mode": "ei",
            "use_gridding": True,
            "use_nonadjacent": True,
            "batch_size": BATCH_SIZE,
        },
    }


# -----------------------------------------------------------------------------
# Seed-level execution helpers
# -----------------------------------------------------------------------------
def generate_initial_points(spec: Dict, seed: int, init_size: int) -> pd.DataFrame:
    rng = random.Random(seed)
    point_ids = rng.sample(spec["candidate_table"]["point_id"].astype(int).tolist(), init_size)
    init_df = spec["candidate_table"].loc[spec["candidate_table"]["point_id"].isin(point_ids)].copy()
    init_df["_order"] = init_df["point_id"].map({pid: idx for idx, pid in enumerate(point_ids)})
    init_df = init_df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return init_df


def _record_eval(
    history_rows: List[Dict],
    suite_name: str,
    spec: Dict,
    method_name: str,
    seed: int,
    cycle_idx: int,
    eval_idx_global: int,
    eval_idx_within_cycle: int,
    point_row: pd.Series,
    y_obs: float,
    best_true_so_far: float,
    best_obs_so_far: float,
    prev_row: Optional[pd.Series],
    noise_cfg: Optional[Dict],
    meta: Dict,
    source_stage: str,
):
    record = {
        "suite_name": suite_name,
        "function_name": spec["name"],
        "function_family": spec["family"],
        "method_name": method_name,
        "seed": seed,
        "noise_mode": (noise_cfg or {}).get("mode", "none"),
        "noise_level": float((noise_cfg or {}).get("level", 0.0)),
        "cycle_idx": cycle_idx,
        "eval_idx_global": eval_idx_global,
        "eval_idx_within_cycle": eval_idx_within_cycle,
        "source_stage": source_stage,
        "point_id": int(point_row["point_id"]),
        "y_true": float(point_row["y_true"]),
        "y_obs": float(y_obs),
        "best_true_so_far": float(best_true_so_far),
        "best_obs_so_far": float(best_obs_so_far),
        "anchor_point_id": meta.get("anchor_point_id"),
        "selected_dim": meta.get("selected_dim"),
        "anchor_acq": meta.get("anchor_acq"),
        "direction_acq": meta.get("direction_acq"),
        "used_gridding": meta.get("used_gridding"),
        "used_nonadjacent": meta.get("used_nonadjacent"),
        "prev_point_id": None if prev_row is None else int(prev_row["point_id"]),
        "changed_dims_count_from_prev": 0,
        "step_distance_l1_default": 0.0,
    }
    for dim_name, idx_name in zip(spec["dim_names"], spec["idx_names"]):
        record[dim_name] = float(point_row[dim_name])
        record[idx_name] = int(point_row[idx_name])
        step_col = f"step_distance_{idx_name}"
        if prev_row is None:
            record[step_col] = 0
        else:
            step = abs(int(point_row[idx_name]) - int(prev_row[idx_name]))
            record[step_col] = int(step)
            record["step_distance_l1_default"] += int(step)
            if step > 0:
                record["changed_dims_count_from_prev"] += 1

    for thr in DEFAULT_THRESHOLDS:
        thr_name = int(round(thr * 100))
        record[f"success_{thr_name}"] = int(best_true_so_far >= thr * spec["global_max_value"])

    history_rows.append(record)


def _summarise_run(history_df: pd.DataFrame, spec: Dict, method_name: str, seed: int, init_size: int, total_budget: int, noise_cfg: Optional[Dict]) -> pd.DataFrame:
    row = {
        "function_name": spec["name"],
        "method_name": method_name,
        "seed": seed,
        "init_size": init_size,
        "total_budget": total_budget,
        "noise_mode": (noise_cfg or {}).get("mode", "none"),
        "noise_level": float((noise_cfg or {}).get("level", 0.0)),
        "final_best_true": float(history_df["best_true_so_far"].iloc[-1]),
        "final_best_obs": float(history_df["best_obs_so_far"].iloc[-1]),
        "global_max_value": float(spec["global_max_value"]),
    }
    for thr in DEFAULT_THRESHOLDS:
        thr_name = int(round(thr * 100))
        success_mask = history_df["best_true_so_far"] >= thr * spec["global_max_value"]
        row[f"success_within_budget_{thr_name}"] = int(bool(success_mask.any()))
        if success_mask.any():
            first_idx = history_df.loc[success_mask].iloc[0]
            row[f"first_success_eval_{thr_name}"] = int(first_idx["eval_idx_global"])
            row[f"first_success_cycle_{thr_name}"] = int(first_idx["cycle_idx"])
        else:
            row[f"first_success_eval_{thr_name}"] = -1
            row[f"first_success_cycle_{thr_name}"] = -1
    return pd.DataFrame([row])


def run_single_seed(
    suite_name: str,
    spec: Dict,
    method_name: str,
    seed: int,
    init_size: int = 15,
    total_budget: int = 100,
    noise_cfg: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    methods = get_method_registry()
    if method_name not in methods:
        raise KeyError(f"Unknown method_name: {method_name}")
    method_cfg = methods[method_name]

    rng = np.random.default_rng(seed)
    init_df = generate_initial_points(spec, seed, init_size)

    observed_rows = []
    history_rows = []
    prev_row = None
    best_true_so_far = -np.inf
    best_obs_so_far = -np.inf
    eval_idx_global = 0

    for init_rank, point_row in enumerate(init_df.itertuples(index=False), start=1):
        point_series = pd.Series(point_row._asdict())
        y_obs = sample_observation(float(point_series["y_true"]), point_series, spec, noise_cfg, rng)
        observed_entry = point_series.to_dict()
        observed_entry["y_obs"] = float(y_obs)
        observed_rows.append(observed_entry)

        eval_idx_global += 1
        best_true_so_far = max(best_true_so_far, float(point_series["y_true"]))
        best_obs_so_far = max(best_obs_so_far, float(y_obs))
        _record_eval(
            history_rows=history_rows,
            suite_name=suite_name,
            spec=spec,
            method_name=method_name,
            seed=seed,
            cycle_idx=0,
            eval_idx_global=eval_idx_global,
            eval_idx_within_cycle=init_rank,
            point_row=point_series,
            y_obs=y_obs,
            best_true_so_far=best_true_so_far,
            best_obs_so_far=best_obs_so_far,
            prev_row=prev_row,
            noise_cfg=noise_cfg,
            meta={
                "anchor_point_id": None,
                "selected_dim": None,
                "anchor_acq": None,
                "direction_acq": None,
                "used_gridding": None,
                "used_nonadjacent": None,
            },
            source_stage="init",
        )
        prev_row = point_series

    observed_df = pd.DataFrame(observed_rows)
    remaining_budget = total_budget - init_size
    if remaining_budget < 0:
        raise ValueError("total_budget must be >= init_size")

    if method_cfg["type"] == "sequential_dragonfly":
        opt = build_dragonfly_seq_optimizer(spec, observed_df)
        observed_ids = set(observed_df["point_id"].astype(int).tolist())
        for step in range(remaining_budget):
            point_series = None
            for _ in range(30):
                x = opt.ask()
                coord = _canonicalize_raw_point(x, spec)
                point_id = spec["coord_to_pid"][coord]
                if point_id not in observed_ids:
                    point_series = spec["candidate_table"].loc[spec["candidate_table"]["point_id"] == point_id].iloc[0]
                    break
            if point_series is None:
                remaining_df = spec["candidate_table"].loc[~spec["candidate_table"]["point_id"].isin(observed_ids)]
                point_series = remaining_df.sample(n=1, random_state=seed + step).iloc[0]
                coord = tuple(point_series[d] for d in spec["dim_names"])
                x = list(coord)

            y_obs = sample_observation(float(point_series["y_true"]), point_series, spec, noise_cfg, rng)
            opt.tell([(list(coord), float(y_obs))])
            observed_ids.add(int(point_series["point_id"]))
            observed_entry = point_series.to_dict()
            observed_entry["y_obs"] = float(y_obs)
            observed_df = pd.concat([observed_df, pd.DataFrame([observed_entry])], ignore_index=True)

            eval_idx_global += 1
            best_true_so_far = max(best_true_so_far, float(point_series["y_true"]))
            best_obs_so_far = max(best_obs_so_far, float(y_obs))
            _record_eval(
                history_rows=history_rows,
                suite_name=suite_name,
                spec=spec,
                method_name=method_name,
                seed=seed,
                cycle_idx=step + 1,
                eval_idx_global=eval_idx_global,
                eval_idx_within_cycle=1,
                point_row=point_series,
                y_obs=y_obs,
                best_true_so_far=best_true_so_far,
                best_obs_so_far=best_obs_so_far,
                prev_row=prev_row,
                noise_cfg=noise_cfg,
                meta={
                    "anchor_point_id": None,
                    "selected_dim": None,
                    "anchor_acq": "dragonfly_default",
                    "direction_acq": None,
                    "used_gridding": False,
                    "used_nonadjacent": False,
                },
                source_stage="adaptive",
            )
            prev_row = point_series

    elif method_cfg["type"] in {"topk_batch", "ofat"}:
        n_cycles = remaining_budget // BATCH_SIZE
        if remaining_budget % BATCH_SIZE != 0:
            raise ValueError("For batch methods, remaining_budget must be divisible by batch size.")

        for cycle in range(1, n_cycles + 1):
            if method_cfg["type"] == "topk_batch":
                selected_df, meta = propose_topk_batch(spec, observed_df, acq_mode=method_cfg["acq_mode"], top_k=BATCH_SIZE)
            else:
                selected_df, meta = propose_ofat_batch(
                    spec,
                    observed_df,
                    anchor_mode=method_cfg["anchor_mode"],
                    direction_mode=method_cfg["direction_mode"],
                    use_gridding=method_cfg["use_gridding"],
                    use_nonadjacent=method_cfg["use_nonadjacent"],
                )

            for batch_rank, point_row in enumerate(selected_df.itertuples(index=False), start=1):
                point_series = pd.Series(point_row._asdict())
                y_obs = sample_observation(float(point_series["y_true"]), point_series, spec, noise_cfg, rng)
                observed_entry = point_series.to_dict()
                observed_entry["y_obs"] = float(y_obs)
                observed_df = pd.concat([observed_df, pd.DataFrame([observed_entry])], ignore_index=True)

                eval_idx_global += 1
                best_true_so_far = max(best_true_so_far, float(point_series["y_true"]))
                best_obs_so_far = max(best_obs_so_far, float(y_obs))
                _record_eval(
                    history_rows=history_rows,
                    suite_name=suite_name,
                    spec=spec,
                    method_name=method_name,
                    seed=seed,
                    cycle_idx=cycle,
                    eval_idx_global=eval_idx_global,
                    eval_idx_within_cycle=batch_rank,
                    point_row=point_series,
                    y_obs=y_obs,
                    best_true_so_far=best_true_so_far,
                    best_obs_so_far=best_obs_so_far,
                    prev_row=prev_row,
                    noise_cfg=noise_cfg,
                    meta=meta,
                    source_stage="adaptive",
                )
                prev_row = point_series
    else:
        raise ValueError(f"Unknown method type: {method_cfg['type']}")

    history_df = pd.DataFrame(history_rows)
    summary_df = _summarise_run(history_df, spec, method_name, seed, init_size, total_budget, noise_cfg)
    return history_df, summary_df


# -----------------------------------------------------------------------------
# Suite-level helpers
# -----------------------------------------------------------------------------
def aggregate_summary(run_summary_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["function_name", "method_name", "noise_mode", "noise_level"]
    rows = []
    for group_key, group_df in run_summary_df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, group_key))
        row["n_seeds"] = int(len(group_df))
        row["mean_final_best_true"] = float(group_df["final_best_true"].mean())
        row["std_final_best_true"] = float(group_df["final_best_true"].std(ddof=0)) if len(group_df) > 1 else 0.0
        for thr in DEFAULT_THRESHOLDS:
            thr_name = int(round(thr * 100))
            success_col = f"success_within_budget_{thr_name}"
            first_eval_col = f"first_success_eval_{thr_name}"
            row[f"success_rate_{thr_name}"] = float(group_df[success_col].mean())
            valid = group_df.loc[group_df[first_eval_col] >= 0, first_eval_col]
            row[f"median_first_success_eval_{thr_name}"] = float(valid.median()) if len(valid) else -1.0
            row[f"p80_first_success_eval_{thr_name}"] = float(valid.quantile(0.8)) if len(valid) else -1.0
            row[f"p90_first_success_eval_{thr_name}"] = float(valid.quantile(0.9)) if len(valid) else -1.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_config_snapshot(path: str, config: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
