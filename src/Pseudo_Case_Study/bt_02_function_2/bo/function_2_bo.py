from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='dragonfly.utils.oper_utils')
from dragonfly import maximise_function, load_config
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def f(T, t2, beta):
    T_k = T + 273.15
    k_forward = 1e9 * np.exp(-48000 / (8.314 * T_k))
    k_reverse = 5e11 * np.exp(-92000 / (8.314 * T_k))
    base_effect = k_forward / (k_forward + k_reverse)
    gaussian = np.exp(-0.5 * ((T - 5) / 8)  ** 2)
    temp_effect = base_effect * gaussian
    k1, k2 = 0.12, 0.08
    time_effect = (k1 / (k2 - k1)) * (np.exp(-k1 * t2) - np.exp(-k2 * t2))
    beta_effect = np.where(beta <= 110, (70 + 30 * (beta - 100) / 10) * 0.01, 1.0)
    total_yield = temp_effect * time_effect * beta_effect
    return ((total_yield - 0.0) / (0.441 - 0.0)) * 60 + 40


def objective(x):
    T = x[0]
    t2 = x[1]
    beta = x[2]
    return f(T, t2, beta)


max_capital = 50
domain_vars = [
    {'name': 'T', 'type': 'discrete_numeric',
     'items': [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]},
    {'name': 't2', 'type': 'discrete_numeric', 'items': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
    {'name': 'beta', 'type': 'discrete_numeric', 'items': [100, 105, 110, 115, 120]}
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

    yields = [q_info.true_val for q_info in history.query_qinfos]
    first_gt_99_index = next((i for i, y in enumerate(yields) if y > 99.99), None)

    if first_gt_99_index is not None:
        max_yield_index = first_gt_99_index
    else:
        max_yield_index = yields.index(max(yields))

    done_index = max_yield_index if max(yields) > 99.99 else -1

    return {
        'seed': seed,
        'max_yield_index': max_yield_index,
        'done_index': done_index,
        'max_yield': max(yields)
    }


if __name__ == '__main__':
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_seed)(seed) for seed in range(2000, 3000)
    )

    done_df = pd.DataFrame(results).sort_values('seed').reset_index(drop=True)

    assert done_df['seed'].equals(pd.Series(range(2000, 3000))), "Sequential verification failed"

    done_df.to_csv('function_2_bo.csv', index=False)
    print("Done. Results sorted and saved.")