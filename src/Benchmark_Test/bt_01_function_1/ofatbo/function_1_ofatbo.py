import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='dragonfly.utils.oper_utils')
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from argparse import Namespace
from dragonfly import load_config
from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.opt.gp_bandit import CPGPBandit
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from scipy.stats import norm
import random
from joblib import Parallel, delayed


def ucb(mu, sigma, kappa=2.576):
    return mu + kappa * sigma


def ei(mu, sigma, tau, xi=0.01):
    with np.errstate(divide='warn'):
        z = (mu - tau - xi) / sigma
        ei_val = (mu - tau - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei_val = np.array([ei_val])
        ei_val[ei_val < 0] = 0
    return ei_val[0]


def beta_acq(df):
    n_data = len(df)
    T = df['T'].values
    t2 = df['t2'].values
    beta = df['beta'].values
    yields = df['yield'].values

    X = []
    Y = []
    for i in range(n_data):
        x = [T[i], t2[i], beta[i]]
        y = [yields[i]]
        X.append(x)
        Y.append(y)

    domain_vars = [
        {'name': 'T', 'type': 'discrete_numeric', 'items': [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]},
        {'name': 't2', 'type': 'discrete_numeric', 'items': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
        {'name': 'beta', 'type': 'discrete_numeric', 'items': [100, 105, 110, 115, 120]}
    ]
    config_params = {'domain': domain_vars}
    config = load_config(config_params)

    num_init = 1
    options = Namespace(gpb_hp_tune_criterion='ml')
    func_caller = CPFunctionCaller(None, config.domain, domain_orderings=config.domain_orderings)
    wm = SyntheticWorkerManager(1)
    opt = CPGPBandit(func_caller, 'default', ask_tell_mode=True, options=options)
    opt.worker_manager = None
    opt._set_up()

    opt.initialise()
    for i in range(n_data):
        x = X[i]
        y = Y[i][0]
        opt.tell([(x, y)])
        opt.step_idx += 1
    opt._build_new_model()
    opt._set_next_gp()

    results = []
    for T_idx, T_val in enumerate(domain_vars[0]['items']):
        for t2_idx, t2_val in enumerate(domain_vars[1]['items']):
            for beta_idx, beta_val in enumerate(domain_vars[2]['items']):
                x_raw = [T_val, t2_val, beta_val]
                x_input = [opt.func_caller.get_processed_domain_point_from_raw(x_raw)]
                mu, stdev = opt.gp.eval(x_input, uncert_form='std')
                results.append([T_val, t2_val, beta_val, mu[0], stdev[0]])

    results_df = pd.DataFrame(results, columns=['T', 't2', 'beta', 'mu', 'stdev'])

    tau = np.max(yields)
    acquisition_df = results_df.copy()
    acquisition_df['value_ucb'] = acquisition_df.apply(lambda row: ucb(row['mu'], row['stdev']), axis=1)
    acquisition_df['value_ei'] = acquisition_df.apply(lambda row: ei(row['mu'], row['stdev'], tau), axis=1)

    existing_points = df[['T', 't2', 'beta']].drop_duplicates()
    existing_tuples = set(existing_points.itertuples(index=False, name=None))
    mask = acquisition_df.apply(lambda row: (row['T'], row['t2'], row['beta']) in existing_tuples, axis=1)
    acquisition_unknown = acquisition_df[~mask]

    ei_max_index = acquisition_unknown['value_ei'].idxmax()
    max_ei_row = acquisition_unknown.loc[ei_max_index]

    value_ucb = acquisition_df['value_ucb'].values
    value_ucb_reshaped = value_ucb.reshape(21, 16, 5)
    filled_data_ucb = np.pad(value_ucb_reshaped, pad_width=((2, 2), (2, 2), (0, 0)), mode='edge')
    result_ucb = np.zeros((5, 5, 5))
    for i in range(5):
        for j in range(5):
            for k in range(5):
                region_ucb = filled_data_ucb[i * 5:(i + 1) * 5, j * 4:(j + 1) * 4, k]
                result_ucb[i, j, k] = np.mean(region_ucb)

    max_ei_T_idx = domain_vars[0]['items'].index(max_ei_row['T'])
    max_ei_t2_idx = domain_vars[1]['items'].index(max_ei_row['t2'])
    max_ei_beta_idx = domain_vars[2]['items'].index(max_ei_row['beta'])
    offset_T = 2
    offset_t2 = 2
    offset_beta = 0
    max_ei_T_idx = int((max_ei_T_idx - offset_T) / 5)
    max_ei_t2_idx = int((max_ei_t2_idx - offset_t2) / 4)
    max_ei_beta_idx = int((max_ei_beta_idx - offset_beta) / 1)
    max_ei_T_idx = max(0, min(max_ei_T_idx, 4))
    max_ei_t2_idx = max(0, min(max_ei_t2_idx, 4))
    max_ei_beta_idx = max(0, min(max_ei_beta_idx, 4))
    selected_rows = max_ei_row.to_frame().T

    mean_x = np.mean(result_ucb[max_ei_T_idx, :, :])
    mean_y = np.mean(result_ucb[:, max_ei_t2_idx, :])
    mean_z = np.mean(result_ucb[:, :, max_ei_beta_idx])

    directions = [
        {'name': 'x', 'mean': mean_x},
        {'name': 'y', 'mean': mean_y},
        {'name': 'z', 'mean': mean_z}
    ]
    sorted_directions = sorted(directions, key=lambda d: -d['mean'])

    max_line_df = None
    for direction in sorted_directions:
        if direction['name'] == 'x':
            temp_df = acquisition_df[
                (acquisition_df['t2'] == max_ei_row['t2']) &
                (acquisition_df['beta'] == max_ei_row['beta'])
                ].sort_values('T').reset_index(drop=True)
        elif direction['name'] == 'y':
            temp_df = acquisition_df[
                (acquisition_df['T'] == max_ei_row['T']) &
                (acquisition_df['beta'] == max_ei_row['beta'])
                ].sort_values('t2').reset_index(drop=True)
        else:
            temp_df = acquisition_df[
                (acquisition_df['T'] == max_ei_row['T']) &
                (acquisition_df['t2'] == max_ei_row['t2'])
                ].sort_values('beta').reset_index(drop=True)

        mask = (
                       (temp_df['T'] == max_ei_row['T']) &
                       (temp_df['t2'] == max_ei_row['t2']) &
                       (temp_df['beta'] == max_ei_row['beta'])
               ) | temp_df.apply(
            lambda row: (row['T'], row['t2'], row['beta']) in existing_tuples,
            axis=1
        )
        filtered_df = temp_df[~mask]

        if len(filtered_df) >= 4:
            max_line_df = temp_df
            break

    if max_line_df is None:
        raise ValueError("No direction has enough points after filtering.")

    mask = (
                   (max_line_df['T'] == max_ei_row['T']) &
                   (max_line_df['t2'] == max_ei_row['t2']) &
                   (max_line_df['beta'] == max_ei_row['beta'])
           ) | max_line_df.apply(
        lambda row: (row['T'], row['t2'], row['beta']) in existing_tuples,
        axis=1
    )
    max_line_df = max_line_df[~mask].reset_index(drop=True)

    for _ in range(4):
        if len(max_line_df) >= 7:
            max_ucb_index = max_line_df['value_ucb'].idxmax()
            max_ucb_row = max_line_df.loc[max_ucb_index]
            current_idx = max_ucb_row.name
            indices_to_drop = [current_idx - 1, current_idx + 1]
            valid_indices = [idx for idx in indices_to_drop if idx in max_line_df.index]
            max_line_df = max_line_df.drop(valid_indices).reset_index(drop=True)

        max_ucb_index = max_line_df['value_ucb'].idxmax()
        max_ucb_row = max_line_df.loc[max_ucb_index]
        selected_rows = pd.concat([selected_rows, max_ucb_row.to_frame().T], ignore_index=True)
        max_line_df = max_line_df.drop(max_ucb_row.name).reset_index(drop=True)

    selected_rows['yield'] = 0
    selected_rows = selected_rows[['T', 't2', 'beta', 'yield']]
    combined_data = pd.concat([df, selected_rows], ignore_index=True)
    return combined_data

T_f_range = np.arange(-20, 21, 2)
t2_f_range = np.arange(5, 21, 1)
beta_f_range = np.arange(100, 121, 5)
lookup_table = {}
for T in T_f_range:
    for t2 in t2_f_range:
        for beta in beta_f_range:
            f_T = 50 + 15 * np.exp(-((T + 5) ** 2) / 6) + 25 * np.exp(-((T - 5) ** 2) / 6)
            f_t2 = 40 + 30 * np.exp(-((t2 - 13) ** 2) / 6)
            f_beta = 70 + 20 * np.exp(-((beta - 115) ** 2) / 30)
            lookup_table[(T, t2, beta)] = (f_T * f_t2 * f_beta) / 4500


def f(T, t2, beta):
    return lookup_table.get((T, t2, beta), None)


def process_seed(args):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='dragonfly.utils.oper_utils')
    seed, n_init, n_total_iter, done_threshold = args
    random.seed(seed)

    T_items = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    t2_items = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    beta_items = [100, 105, 110, 115, 120]

    T_selected = random.sample(T_items, n_init) if n_init <= len(T_items) else random.choices(T_items, k=n_init)
    t2_selected = random.sample(t2_items, n_init) if n_init <= len(t2_items) else random.choices(t2_items, k=n_init)
    beta_selected = random.sample(beta_items, n_init) if n_init <= len(beta_items) else random.choices(beta_items,
                                                                                                       k=n_init)

    df = pd.DataFrame({
        'T': T_selected,
        't2': t2_selected,
        'beta': beta_selected
    })
    df['yield'] = df.apply(lambda row: f(row['T'], row['t2'], row['beta']), axis=1)

    n_iter = 0
    while n_iter < n_total_iter:
        df = beta_acq(df)

        last_five = df.tail(5)
        for idx, row in last_five.iterrows():
            new_yield = f(row['T'], row['t2'], row['beta'])
            df.at[idx, 'yield'] = new_yield

        n_iter += 5

    done_yield_rows = df[df['yield'] > done_threshold]
    if not done_yield_rows.empty:
        max_yield_index = done_yield_rows.index.min()
        done_index = done_yield_rows.index.min()
        max_yield = df.loc[max_yield_index, 'yield']
    else:
        max_yield_index = df['yield'].idxmax()
        max_yield = df.loc[max_yield_index, 'yield']
        done_index = -1

    return pd.DataFrame({
        'seed': [seed],
        'max_yield_index': [max_yield_index],
        'done_index': [done_index],
        'max_yield': [max_yield]
    })


if __name__ == '__main__':
    n_seed = 20
    done_threshold = 99

    n_init = 15
    n_total_iter = 100 - n_init

    seeds = range(2000, 2000 + n_seed)
    args_list = [(seed, n_init, n_total_iter, done_threshold) for seed in seeds]

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_seed)(args) for args in args_list
    )

    done_df = pd.concat(results).sort_values('seed').reset_index(drop=True)

    print(f"\nDone!")
    done_df.to_csv(
        f'function_1_betabo.csv',
        index=False
    )