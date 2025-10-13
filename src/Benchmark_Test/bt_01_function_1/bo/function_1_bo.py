from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='dragonfly.utils.oper_utils')
from dragonfly import maximise_function, load_config
import numpy as np
import pandas as pd


def f(T, t2, beta):
    f_T = 50 + 15 * np.exp(-((T + 5)  **  2) / 6) + 25 * np.exp(-((T - 5)  **  2) / 6)
    f_t2 = 40 + 30 * np.exp(-((t2 - 13)  **  2) / 6)
    f_beta = 70 + 20 * np.exp(-((beta - 115)  **  2) / 30)
    return (f_T * f_t2 * f_beta) / 4500


def objective(x):
    T = x[0]
    t2 = x[1]
    beta = x[2]
    return f(T, t2, beta)


max_capital = 50
domain_vars = [
    {
        'name': 'T',
        'type': 'discrete_numeric',
        'items': [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    },
    {
        'name': 't2',
        'type': 'discrete_numeric',
        'items': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    },
    {
        'name': 'beta',
        'type': 'discrete_numeric',
        'items': [100, 105, 110, 115, 120]
    }
]
config_params = {'domain': domain_vars}
config = load_config(config_params)

done_df = pd.DataFrame(columns=['seed', 'max_yield_index', 'done_index', 'max_yield'])

for seed in range(2000, 3000):
    np.random.seed(seed)

    opt_val, opt_pt, history = maximise_function(func=objective,
                                                 domain=config.domain,
                                                 max_capital=max_capital,
                                                 config=config,
                                                 reporter='silent'
                                                 )

    yields = [q_info.true_val for q_info in history.query_qinfos]

    first_gt_99_index = next((i for i, y in enumerate(yields) if y > 99), None)

    if first_gt_99_index is not None:
        max_yield_index = first_gt_99_index
    else:
        max_yield_index = yields.index(max(yields))

    done_index = max_yield_index if max(yields) > 99 else -1

    done_df = pd.concat([done_df, pd.DataFrame({
        'seed': [seed],
        'max_yield_index': [max_yield_index],
        'done_index': [done_index],
        'max_yield': [max(yields)]
    })], ignore_index=True)
    print(f"Seed {seed} done.")

done_df.to_csv('function_1_bo.csv', index=False)