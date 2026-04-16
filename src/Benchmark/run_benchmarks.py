import argparse
import os
from typing import Dict, List, Optional

import pandas as pd
from joblib import Parallel, delayed

from benchmark_harness import (
    aggregate_summary,
    build_function_registry,
    get_method_registry,
    run_single_seed,
    save_config_snapshot,
)


DEFAULT_SUITE_CONFIGS = {
    "screening": {
        "functions": [
            "hartmann3",
            "ackley3",
            "branin_weak_z",
            "hetero_cardinality_stress",
            "rotated_ridge",
            "rotated_rosenbrock",
        ],
        "methods": [
            "seq_dragonfly",
            "batch_ei_top5",
            "batch_ucb_top5",
            "ofat_full",
        ],
        "n_seeds": 100,
        "noise_cfgs": [{"mode": "none", "level": 0.0}],
    },
    "main": {
        # Placeholder defaults; after screening, you can override with --functions.
        "functions": [
            "hartmann3",
            "hetero_cardinality_stress",
            "rotated_ridge",
        ],
        "methods": [
            "seq_dragonfly",
            "batch_ei_top5",
            "batch_ucb_top5",
            "ofat_full",
        ],
        "n_seeds": 1000,
        "noise_cfgs": [{"mode": "none", "level": 0.0}],
    },
    "ablation": {
        "functions": [
            "hetero_cardinality_stress",
            "rotated_ridge",
        ],
        "methods": [
            "ofat_full",
            "ofat_no_gridding",
            "ofat_no_nonadjacent",
            "ofat_ucb_ucb",
            "ofat_ei_ei",
            "batch_ei_top5",
        ],
        "n_seeds": 200,
        "noise_cfgs": [{"mode": "none", "level": 0.0}],
    },
    "noise": {
        "functions": [
            "hartmann3",
            "hetero_cardinality_stress",
        ],
        "methods": [
            "seq_dragonfly",
            "batch_ei_top5",
            "batch_ucb_top5",
            "ofat_full",
        ],
        "n_seeds": 200,
        "noise_cfgs": [
            {"mode": "additive_gaussian", "level": 0.01},
            {"mode": "additive_gaussian", "level": 0.03},
            {"mode": "additive_gaussian", "level": 0.05},
            {"mode": "heteroscedastic_gaussian", "level": 0.03},
        ],
    },
}


def parse_csv_arg(text: Optional[str]) -> Optional[List[str]]:
    if text is None or text.strip() == "":
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def get_noise_tag(noise_cfg: Dict) -> str:
    mode = noise_cfg.get("mode", "none")
    level = float(noise_cfg.get("level", 0.0))
    if mode == "none":
        return "noise_none"
    return f"noise_{mode}_{str(level).replace('.', 'p')}"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_combo(
    suite_name: str,
    function_name: str,
    method_name: str,
    noise_cfg: Dict,
    n_seeds: int,
    seed_start: int,
    init_size: int,
    total_budget: int,
    n_jobs: int,
    output_dir: str,
):
    registry = build_function_registry()
    spec = registry[function_name]
    seeds = list(range(seed_start, seed_start + n_seeds))

    combo_dir = os.path.join(output_dir, suite_name, function_name, method_name, get_noise_tag(noise_cfg))
    ensure_dir(combo_dir)

    print(
        f"Running suite={suite_name} | function={function_name} | method={method_name} | "
        f"noise={noise_cfg.get('mode', 'none')} ({noise_cfg.get('level', 0.0)}) | seeds={n_seeds}"
    )

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_seed)(
            suite_name=suite_name,
            spec=spec,
            method_name=method_name,
            seed=seed,
            init_size=init_size,
            total_budget=total_budget,
            noise_cfg=noise_cfg,
        )
        for seed in seeds
    )

    histories = [res[0] for res in results]
    summaries = [res[1] for res in results]

    history_df = pd.concat(histories, ignore_index=True)
    summary_df = pd.concat(summaries, ignore_index=True)
    agg_df = aggregate_summary(summary_df)

    history_df.to_csv(os.path.join(combo_dir, "run_history.csv"), index=False)
    summary_df.to_csv(os.path.join(combo_dir, "run_summary.csv"), index=False)
    agg_df.to_csv(os.path.join(combo_dir, "aggregate_summary.csv"), index=False)

    function_metadata = {
        "function_name": function_name,
        "function_family": spec["family"],
        "dim_names": spec["dim_names"],
        "cardinalities": [len(v) for v in spec["levels"]],
        "global_max_value": spec["global_max_value"],
        "global_max_point_ids": spec["global_max_point_ids"],
        "n_candidates": int(len(spec["candidate_table"])),
    }
    save_config_snapshot(os.path.join(combo_dir, "function_metadata.json"), function_metadata)

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Run unified benchmark harness for OFAT revision experiments.")
    parser.add_argument("--suite", choices=["screening", "main", "ablation", "noise"], required=True)
    parser.add_argument("--functions", type=str, default=None, help="Comma-separated function names to override defaults.")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method names to override defaults.")
    parser.add_argument("--n-seeds", type=int, default=None, help="Override default number of seeds for the suite.")
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--init-size", type=int, default=15)
    parser.add_argument("--total-budget", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    registry = build_function_registry()
    method_registry = get_method_registry()
    suite_cfg = DEFAULT_SUITE_CONFIGS[args.suite].copy()

    functions = parse_csv_arg(args.functions) or suite_cfg["functions"]
    methods = parse_csv_arg(args.methods) or suite_cfg["methods"]
    n_seeds = int(args.n_seeds) if args.n_seeds is not None else int(suite_cfg["n_seeds"])
    noise_cfgs = suite_cfg["noise_cfgs"]

    unknown_functions = [f for f in functions if f not in registry]
    unknown_methods = [m for m in methods if m not in method_registry]
    if unknown_functions:
        raise ValueError(f"Unknown function names: {unknown_functions}")
    if unknown_methods:
        raise ValueError(f"Unknown method names: {unknown_methods}")

    ensure_dir(args.output_dir)
    suite_root = os.path.join(args.output_dir, args.suite)
    ensure_dir(suite_root)

    snapshot = {
        "suite": args.suite,
        "functions": functions,
        "methods": methods,
        "n_seeds": n_seeds,
        "seed_start": args.seed_start,
        "init_size": args.init_size,
        "total_budget": args.total_budget,
        "noise_cfgs": noise_cfgs,
    }
    save_config_snapshot(os.path.join(suite_root, "suite_config.json"), snapshot)

    all_summaries = []
    for function_name in functions:
        for method_name in methods:
            for noise_cfg in noise_cfgs:
                summary_df = run_combo(
                    suite_name=args.suite,
                    function_name=function_name,
                    method_name=method_name,
                    noise_cfg=noise_cfg,
                    n_seeds=n_seeds,
                    seed_start=args.seed_start,
                    init_size=args.init_size,
                    total_budget=args.total_budget,
                    n_jobs=args.n_jobs,
                    output_dir=args.output_dir,
                )
                all_summaries.append(summary_df)

    all_summaries_df = pd.concat(all_summaries, ignore_index=True)
    all_summaries_df.to_csv(os.path.join(suite_root, "all_run_summaries.csv"), index=False)
    aggregate_summary(all_summaries_df).to_csv(os.path.join(suite_root, "all_aggregate_summary.csv"), index=False)
    print(f"Done. Results written to: {os.path.abspath(suite_root)}")


if __name__ == "__main__":
    main()
