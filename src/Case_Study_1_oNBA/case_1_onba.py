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
    c = df['c'].values
    t = df['t'].values
    p = df['p'].values
    xc = df['xc'].values

    X = []
    Y = []
    for i in range(n_data):
        x = [c[i], t[i], p[i]]
        y = [xc[i]]
        X.append(x)
        Y.append(y)

    domain_vars = [
        {'name': 'c', 'type': 'discrete_numeric', 'items': [40, 60, 80, 100, 120, 140, 160, 180, 200]},
        {'name': 't', 'type': 'discrete_numeric', 'items': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {'name': 'p', 'type': 'discrete_numeric', 'items': [18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0]}
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
    for c_idx, c_val in enumerate(domain_vars[0]['items']):
        for t_idx, t_val in enumerate(domain_vars[1]['items']):
            for p_idx, p_val in enumerate(domain_vars[2]['items']):
                x_raw = [c_val, t_val, p_val]
                x_input = [opt.func_caller.get_processed_domain_point_from_raw(x_raw)]
                mu, stdev = opt.gp.eval(x_input, uncert_form='std')
                results.append([c_val, t_val, p_val, mu[0], stdev[0]])

    results_df = pd.DataFrame(results, columns=['c', 't', 'p', 'mu', 'stdev'])

    tau = np.max(xc)
    acquisition_df = results_df.copy()
    acquisition_df['value_ucb'] = acquisition_df.apply(lambda row: ucb(row['mu'], row['stdev']), axis=1)
    acquisition_df['value_ei'] = acquisition_df.apply(lambda row: ei(row['mu'], row['stdev'], tau), axis=1)

    existing_points = df[['c', 't', 'p']].drop_duplicates()
    existing_tuples = set(existing_points.itertuples(index=False, name=None))
    mask = acquisition_df.apply(lambda row: (row['c'], row['t'], row['p']) in existing_tuples, axis=1)
    acquisition_unknown = acquisition_df[~mask]

    ei_max_index = acquisition_unknown['value_ei'].idxmax()
    max_ei_row = acquisition_unknown.loc[ei_max_index]

    selected_rows = max_ei_row.to_frame().T
    mean_x = acquisition_df.loc[(acquisition_df['t'] == max_ei_row['t']) & (acquisition_df['p'] == max_ei_row['p']), 'value_ucb'].mean()
    mean_y = acquisition_df.loc[(acquisition_df['c'] == max_ei_row['c']) & (acquisition_df['p'] == max_ei_row['p']), 'value_ucb'].mean()
    mean_z = acquisition_df.loc[(acquisition_df['c'] == max_ei_row['c']) & (acquisition_df['t'] == max_ei_row['t']), 'value_ucb'].mean()

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
                (acquisition_df['t'] == max_ei_row['t']) &
                (acquisition_df['p'] == max_ei_row['p'])
                ].sort_values('c').reset_index(drop=True)
        elif direction['name'] == 'y':
            temp_df = acquisition_df[
                (acquisition_df['c'] == max_ei_row['c']) &
                (acquisition_df['p'] == max_ei_row['p'])
                ].sort_values('t').reset_index(drop=True)
        else:
            temp_df = acquisition_df[
                (acquisition_df['c'] == max_ei_row['c']) &
                (acquisition_df['t'] == max_ei_row['t'])
                ].sort_values('p').reset_index(drop=True)

        mask = (
                       (temp_df['c'] == max_ei_row['c']) &
                       (temp_df['t'] == max_ei_row['t']) &
                       (temp_df['p'] == max_ei_row['p'])
               ) | temp_df.apply(
            lambda row: (row['c'], row['t'], row['p']) in existing_tuples,
            axis=1
        )
        filtered_df = temp_df[~mask]

        if len(filtered_df) >= 4:
            max_line_df = temp_df
            break

    if max_line_df is None:
        raise ValueError("No direction has enough points after filtering.")

    mask = (
                   (max_line_df['c'] == max_ei_row['c']) &
                   (max_line_df['t'] == max_ei_row['t']) &
                   (max_line_df['p'] == max_ei_row['p'])
           ) | max_line_df.apply(
        lambda row: (row['c'], row['t'], row['p']) in existing_tuples,
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

    selected_rows['xc'] = 0
    selected_rows = selected_rows[['c', 't', 'p', 'xc']]
    combined_data = pd.concat([df, selected_rows], ignore_index=True)
    return combined_data


def Qa(c, t, csol, V):
    return round((c / csol) * (V / t), 2)


def Qb(t, V, Qa):
    return round((V / t) - Qa, 2)


def tw(t, V, Vtotal):
    return round(t * 1.5 * (Vtotal/V), 2)


csv_name = "case_1_onba.csv"
# mode="initialization"
# mode="iteration"
mode="mu_stdev_extraction"
csol = 200  # mmol/L
V = 6.67  # mL
Vtotal = 7.5  # mL
if mode == "initialization":
    seed = 2259
    n_init = 15
    random.seed(seed)

    c_items = [40, 60, 80, 100, 120, 140, 160, 180, 200]
    t_items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p_items = [18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0]

    c_selected = random.sample(c_items, n_init) if n_init <= len(c_items) else random.choices(c_items, k=n_init)
    t_selected = random.sample(t_items, n_init) if n_init <= len(t_items) else random.choices(t_items, k=n_init)
    p_selected = random.sample(p_items, n_init) if n_init <= len(p_items) else random.choices(p_items, k=n_init)

    df = pd.DataFrame({
        'c': c_selected,
        't': t_selected,
        'p': p_selected
    })
    df['xc'] = df.apply(lambda row: 0, axis=1)

    df['Qa'] = df.apply(lambda row: Qa(row['c'], row['t'], csol, V), axis=1)
    df['Qb'] = df.apply(lambda row: Qb(row['t'], V, row['Qa']), axis=1)
    df['tw'] = df.apply(lambda row: tw(row['t'], V, Vtotal), axis=1)

    df.to_csv(csv_name, index=False)
    df.to_csv(f"backup/initialization_{seed}.csv", index=False)

    print('Initialization finished')

elif mode == "iteration":
    df = pd.read_csv(csv_name)

    df = beta_acq(df)
    df.loc[df.index[-5:], 'xc'] = 0

    df.loc[df.index[-5:], 'Qa'] = df.iloc[-5:].apply(lambda row: Qa(row['c'], row['t'], csol, V), axis=1)
    df.loc[df.index[-5:], 'Qb'] = df.iloc[-5:].apply(lambda row: Qb(row['t'], V, row['Qa']), axis=1)
    df.loc[df.index[-5:], 'tw'] = df.iloc[-5:].apply(lambda row: tw(row['t'], V, Vtotal), axis=1)

    df.to_csv(csv_name, index=False)
    r_num = len(df)
    df.to_csv(f"backup/iteration_{r_num}.csv", index=False)

    print('Iteration finished')

elif mode == "mu_stdev_extraction":
    snar_data = pd.read_csv(csv_name)
    n_data = len(snar_data) -5

    c = snar_data['c'].iloc[:-5].values
    t = snar_data['t'].iloc[:-5].values
    p = snar_data['p'].iloc[:-5].values
    xc = snar_data['xc'].iloc[:-5].values

    X = []
    Y = []
    for i in range(n_data):
        x = [c[i], t[i], p[i]]
        y = [xc[i]]
        X.append(x)
        Y.append(y)

    domain_vars = [
        {
            'name': 'c',
            'type': 'discrete_numeric',
            'items': [40, 60, 80, 100, 120, 140, 160, 180, 200]
        },
        {
            'name': 't',
            'type': 'discrete_numeric',
            'items': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
        {
            'name': 'p',
            'type': 'discrete_numeric',
            'items': [18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0]
        }
    ]
    config_params = {'domain': domain_vars}
    config = load_config(config_params)

    num_init = 1
    options = Namespace(
        gpb_hp_tune_criterion='ml'
    )

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
    for c_val in domain_vars[0]['items']:
        for t_val in domain_vars[1]['items']:
            for p_val in domain_vars[2]['items']:
                x_raw = [c_val, t_val, p_val]
                x_input = [opt.func_caller.get_processed_domain_point_from_raw(x_raw)]
                mu, stdev = opt.gp.eval(x_input, uncert_form='std')
                results.append([c_val, t_val, p_val, mu[0], stdev[0]])

    results_df = pd.DataFrame(results, columns=['c', 't', 'p', 'mu', 'stdev'])

    results_df.to_csv(f'mu_stdev_extraction/extraction_{n_data}.csv', index=False, header=False)

    stdev_reversed = results_df['stdev'].max() - results_df['stdev']
    stdev_normalized = (stdev_reversed - stdev_reversed.min()) / (stdev_reversed.max() - stdev_reversed.min())

    norm = Normalize(vmin=0, vmax=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'oNBA_mu_stdev_extraction_{n_data}points')

    scatter = ax.scatter(results_df['c'], results_df['t'], results_df['p'],
                         c=results_df['mu'], cmap='autumn_r', s=stdev_normalized * 60 + 6, norm=norm)

    fig.colorbar(scatter)

    ax.set_xlabel('c')
    ax.set_ylabel('t')
    ax.set_zlabel('p')

    ax.view_init(elev=30, azim=45)

    ax.set_xlim(40, 200)
    ax.set_ylim(1, 10)
    ax.set_zlim(18.5, 24.0)

    x_ticks = range(40, 200 + 1, 40)
    y_ticks = range(1, 10 + 1, 1)
    z_ticks = [18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)

    ax.text2D(-0.25, -0.1, f"color: mu\n"
                           f"size: stdev(bigger~lower)", transform=ax.transAxes)

    plt.show()

else:
    raise ValueError("Invalid mode. Please choose 'initialization' or 'iteration'.")