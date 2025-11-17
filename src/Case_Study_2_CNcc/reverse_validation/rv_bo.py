from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='dragonfly.utils.oper_utils')
from dragonfly import maximise_function, load_config
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


extraction_df = pd.read_csv(fr'rv_xgboost_ex.csv')


def objective(x):
    t_val, p_val, cat_val, arbr_val, pyrr_val = x

    match = extraction_df[
        (extraction_df['t'] == t_val) &
        (extraction_df['p'] == p_val) &
        (extraction_df['cat'] == cat_val) &
        (extraction_df['arbr'] == arbr_val) &
        (extraction_df['pyrr'] == pyrr_val)
        ]

    if len(match) == 0:
        print(f"Warning: No match found for point {x}")
        return 0.0

    return float(match['yields'].iloc[0])


max_capital = 50

domain_vars = [
    {'name': 't', 'type': 'discrete_numeric', 'items': [5, 6, 7, 8, 9, 10, 11, 12]},
    {'name': 'p', 'type': 'discrete_numeric', 'items': [19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0]},
    {'name': 'cat', 'type': 'discrete_numeric', 'items': [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.00010]},
    {'name': 'arbr', 'type': 'discrete_numeric', 'items': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
    {'name': 'pyrr', 'type': 'discrete_numeric', 'items': [1.0, 1.25, 1.5, 1.75, 2.0]}
]

config_params = {'domain': domain_vars}
config = load_config(config_params)


def process_seed(seed):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='dragonfly.utils.oper_utils')
    np.random.seed(seed)

    opt_val, opt_pt, history = maximise_function(
        func=objective,
        domain=config.domain,
        max_capital=max_capital,
        config=config,
        reporter='silent'
    )

    done_threshold = 0.80
    yields = [q_info.true_val for q_info in history.query_qinfos]
    first_gt_threshold_index = next((i for i, y in enumerate(yields) if y > done_threshold), None)

    if first_gt_threshold_index is not None:
        max_yield_index = first_gt_threshold_index
    else:
        max_yield_index = yields.index(max(yields))

    done_index = max_yield_index if max(yields) > done_threshold else -1

    return {
        'seed': seed,
        'max_yield_index': max_yield_index,
        'done_index': done_index,
        'max_yield': max(yields)
    }


if __name__ == '__main__':
    n_seed = 1000
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_seed)(seed) for seed in range(2000, 2000 + n_seed)
    )

    done_df = pd.DataFrame(results).sort_values('seed').reset_index(drop=True)

    assert done_df['seed'].equals(pd.Series(range(2000, 2000 + n_seed))), ""

    done_df.to_csv('rv_bo.csv', index=False)
