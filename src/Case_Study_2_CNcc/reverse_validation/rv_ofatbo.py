import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='dragonfly.utils.oper_utils')
import pandas as pd
import numpy as np
import itertools
import xgboost as xgb
from dragonfly import maximise_function
from dragonfly.opt.gp_bandit import EuclideanGPBandit
from dragonfly.exd.experiment_caller import EuclideanFunctionCaller
from dragonfly.exd.domains import EuclideanDomain
from dragonfly.exd.worker_manager import SyntheticWorkerManager
import os
import random
from joblib import Parallel, delayed
from argparse import Namespace
from dragonfly import load_config
from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.opt.gp_bandit import CPGPBandit
from scipy.stats import norm
import imageio.v2 as imageio
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

if not os.path.exists('csv'):
    os.makedirs('csv')

experiment_data = pd.read_csv("experiment_data.csv")

PARAM_BOUNDS = {
    'n_estimators': (80, 150),
    'max_depth': (4, 8),
    'learning_rate': (0.05, 0.2),
    'subsample': (0.7, 0.9),
    'colsample_bytree': (0.8, 1.0),
    'reg_alpha': (0, 0.1),
    'reg_lambda': (0.8, 1.5)
}


def train_xgboost_and_predict(xgboost_params):
    t_items = [5, 6, 7, 8, 9, 10, 11, 12]
    p_items = [19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0]
    cat_items = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005,
                 0.00006, 0.00007, 0.00008, 0.00009, 0.00010]
    arbr_items = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    pyrr_items = [1.0, 1.25, 1.5, 1.75, 2.0]

    feature_cols = ['t', 'p', 'cat', 'arbr', 'pyrr']
    X_train = experiment_data[feature_cols].values
    y_train = experiment_data['yields'].values

    model = xgb.XGBRegressor(
        n_estimators=int(xgboost_params['n_estimators']),
        max_depth=int(xgboost_params['max_depth']),
        learning_rate=xgboost_params['learning_rate'],
        subsample=xgboost_params['subsample'],
        colsample_bytree=xgboost_params['colsample_bytree'],
        reg_alpha=xgboost_params['reg_alpha'],
        reg_lambda=xgboost_params['reg_lambda'],
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)

    full_grid = list(itertools.product(t_items, p_items, cat_items, arbr_items, pyrr_items))
    df_full = pd.DataFrame(full_grid, columns=feature_cols)

    y_pred = model.predict(df_full.values)
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)

    df_full['yields'] = y_pred


    return df_full


def ucb(mu, sigma, kappa=2.576):
    return mu + kappa * sigma


def ei(mu, sigma, tau, xi=0.01):
    with np.errstate(divide='warn', invalid='warn'):
        sigma = max(sigma, 1e-10)
        z = (mu - tau - xi) / sigma
        ei_val = (mu - tau - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei_val = np.array([ei_val])
        ei_val[ei_val < 0] = 0
        if np.isnan(ei_val[0]):
            return 0
    return ei_val[0]


def beta_acq(df, extraction_df):
    n_data = len(df)
    t = df['t'].values
    p = df['p'].values
    cat = df['cat'].values
    arbr = df['arbr'].values
    pyrr = df['pyrr'].values
    yields = df['yields'].values

    X = []
    Y = []
    for i in range(n_data):
        x = [t[i], p[i], cat[i], arbr[i], pyrr[i]]
        y = [yields[i]]
        X.append(x)
        Y.append(y)

    domain_vars = [
        {'name': 't', 'type': 'discrete_numeric', 'items': [5, 6, 7, 8, 9, 10, 11, 12]},
        {'name': 'p', 'type': 'discrete_numeric',
         'items': [19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0]},
        {'name': 'cat', 'type': 'discrete_numeric',
         'items': [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.00010]},
        {'name': 'arbr', 'type': 'discrete_numeric', 'items': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
        {'name': 'pyrr', 'type': 'discrete_numeric', 'items': [1.0, 1.25, 1.5, 1.75, 2.0]}
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

    for t_idx, t_val in enumerate(domain_vars[0]['items']):
        for p_idx, p_val in enumerate(domain_vars[1]['items']):
            for cat_idx, cat_val in enumerate(domain_vars[2]['items']):
                for arbr_idx, arbr_val in enumerate(domain_vars[3]['items']):
                    for pyrr_idx, pyrr_val in enumerate(domain_vars[4]['items']):
                        x_raw = [t_val, p_val, cat_val, arbr_val, pyrr_val]
                        x_input = [opt.func_caller.get_processed_domain_point_from_raw(x_raw)]
                        mu, stdev = opt.gp.eval(x_input, uncert_form='std')
                        results.append([t_val, p_val, cat_val, arbr_val, pyrr_val, mu[0], stdev[0]])

    results_df = pd.DataFrame(results, columns=['t', 'p', 'cat', 'arbr', 'pyrr', 'mu', 'stdev'])

    tau = np.max(yields)
    acquisition_df = results_df.copy()
    acquisition_df['value_ucb'] = acquisition_df.apply(lambda row: ucb(row['mu'], row['stdev']), axis=1)
    acquisition_df['value_ei'] = acquisition_df.apply(lambda row: ei(row['mu'], row['stdev'], tau), axis=1)

    existing_points = df[['t', 'p', 'cat', 'arbr', 'pyrr']].drop_duplicates()
    existing_tuples = set(existing_points.itertuples(index=False, name=None))
    mask = acquisition_df.apply(
        lambda row: (row['t'], row['p'], row['cat'], row['arbr'], row['pyrr']) in existing_tuples, axis=1)
    acquisition_unknown = acquisition_df[~mask]

    ei_max_index = acquisition_unknown['value_ei'].idxmax()
    max_ei_row = acquisition_unknown.loc[ei_max_index]

    selected_rows = max_ei_row.to_frame().T

    mean_dimension_t = acquisition_df.loc[
        (acquisition_df['p'] == max_ei_row['p']) & (acquisition_df['cat'] == max_ei_row['cat']) & (
                acquisition_df['arbr'] == max_ei_row['arbr']) & (
                acquisition_df['pyrr'] == max_ei_row['pyrr']), 'value_ucb'].mean()
    mean_dimension_p = acquisition_df.loc[
        (acquisition_df['t'] == max_ei_row['t']) & (acquisition_df['cat'] == max_ei_row['cat']) & (
                acquisition_df['arbr'] == max_ei_row['arbr']) & (
                acquisition_df['pyrr'] == max_ei_row['pyrr']), 'value_ucb'].mean()
    mean_dimension_cat = acquisition_df.loc[
        (acquisition_df['t'] == max_ei_row['t']) & (acquisition_df['p'] == max_ei_row['p']) & (
                acquisition_df['arbr'] == max_ei_row['arbr']) & (
                acquisition_df['pyrr'] == max_ei_row['pyrr']), 'value_ucb'].mean()
    mean_dimension_arbr = acquisition_df.loc[
        (acquisition_df['t'] == max_ei_row['t']) & (acquisition_df['p'] == max_ei_row['p']) & (
                acquisition_df['cat'] == max_ei_row['cat']) & (
                acquisition_df['pyrr'] == max_ei_row['pyrr']), 'value_ucb'].mean()
    mean_dimension_pyrr = acquisition_df.loc[
        (acquisition_df['t'] == max_ei_row['t']) & (acquisition_df['p'] == max_ei_row['p']) & (
                acquisition_df['cat'] == max_ei_row['cat']) & (
                acquisition_df['arbr'] == max_ei_row['arbr']), 'value_ucb'].mean()

    directions = [
        {'name': 't', 'mean': mean_dimension_t},
        {'name': 'p', 'mean': mean_dimension_p},
        {'name': 'cat', 'mean': mean_dimension_cat},
        {'name': 'arbr', 'mean': mean_dimension_arbr},
        {'name': 'pyrr', 'mean': mean_dimension_pyrr}
    ]
    sorted_directions = sorted(directions, key=lambda d: -d['mean'])

    max_line_df = None
    for direction in sorted_directions:
        if direction['name'] == 't':
            temp_df = acquisition_df[
                (acquisition_df['p'] == max_ei_row['p']) &
                (acquisition_df['cat'] == max_ei_row['cat']) &
                (acquisition_df['arbr'] == max_ei_row['arbr']) &
                (acquisition_df['pyrr'] == max_ei_row['pyrr'])
                ].sort_values('t').reset_index(drop=True)
        elif direction['name'] == 'p':
            temp_df = acquisition_df[
                (acquisition_df['t'] == max_ei_row['t']) &
                (acquisition_df['cat'] == max_ei_row['cat']) &
                (acquisition_df['arbr'] == max_ei_row['arbr']) &
                (acquisition_df['pyrr'] == max_ei_row['pyrr'])
                ].sort_values('p').reset_index(drop=True)
        elif direction['name'] == 'cat':
            temp_df = acquisition_df[
                (acquisition_df['t'] == max_ei_row['t']) &
                (acquisition_df['p'] == max_ei_row['p']) &
                (acquisition_df['arbr'] == max_ei_row['arbr']) &
                (acquisition_df['pyrr'] == max_ei_row['pyrr'])
                ].sort_values('cat').reset_index(drop=True)
        elif direction['name'] == 'arbr':
            temp_df = acquisition_df[
                (acquisition_df['t'] == max_ei_row['t']) &
                (acquisition_df['p'] == max_ei_row['p']) &
                (acquisition_df['cat'] == max_ei_row['cat']) &
                (acquisition_df['pyrr'] == max_ei_row['pyrr'])
                ].sort_values('arbr').reset_index(drop=True)
        else:
            temp_df = acquisition_df[
                (acquisition_df['t'] == max_ei_row['t']) &
                (acquisition_df['p'] == max_ei_row['p']) &
                (acquisition_df['cat'] == max_ei_row['cat']) &
                (acquisition_df['arbr'] == max_ei_row['arbr'])
                ].sort_values('pyrr').reset_index(drop=True)

        mask = (
                       (temp_df['t'] == max_ei_row['t']) &
                       (temp_df['p'] == max_ei_row['p']) &
                       (temp_df['cat'] == max_ei_row['cat']) &
                       (temp_df['arbr'] == max_ei_row['arbr']) &
                       (temp_df['pyrr'] == max_ei_row['pyrr'])
               ) | temp_df.apply(
            lambda row: (row['t'], row['p'], row['cat'], row['arbr'], row['pyrr']) in existing_tuples,
            axis=1
        )
        filtered_df = temp_df[~mask]

        if len(filtered_df) >= 4:
            max_line_df = temp_df
            break

    if max_line_df is None:
        raise ValueError("No direction has enough points after filtering.")

    mask = (
                   (max_line_df['t'] == max_ei_row['t']) &
                   (max_line_df['p'] == max_ei_row['p']) &
                   (max_line_df['cat'] == max_ei_row['cat']) &
                   (max_line_df['arbr'] == max_ei_row['arbr']) &
                   (max_line_df['pyrr'] == max_ei_row['pyrr'])
           ) | max_line_df.apply(
        lambda row: (row['t'], row['p'], row['cat'], row['arbr'], row['pyrr']) in existing_tuples,
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

    selected_rows['yields'] = 0
    selected_rows = selected_rows[['t', 'p', 'cat', 'arbr', 'pyrr', 'yields']]
    combined_data = pd.concat([df, selected_rows], ignore_index=True)
    return combined_data


def objective_function(extraction_df, x):
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


def process_seed(args):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='dragonfly.utils.oper_utils')

    seed, n_init, n_total_iter, done_threshold, extraction_df = args
    random.seed(seed)

    t_items = [5, 6, 7, 8, 9, 10, 11, 12]
    p_items = [19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0]
    cat_items = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.00010]
    arbr_items = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    pyrr_items = [1.0, 1.25, 1.5, 1.75, 2.0]

    t_selected = random.sample(t_items, n_init) if n_init <= len(t_items) else random.choices(t_items, k=n_init)
    p_selected = random.sample(p_items, n_init) if n_init <= len(p_items) else random.choices(p_items, k=n_init)
    cat_selected = random.sample(cat_items, n_init) if n_init <= len(cat_items) else random.choices(cat_items, k=n_init)
    arbr_selected = random.sample(arbr_items, n_init) if n_init <= len(arbr_items) else random.choices(arbr_items,
                                                                                                       k=n_init)
    pyrr_selected = random.sample(pyrr_items, n_init) if n_init <= len(pyrr_items) else random.choices(pyrr_items,
                                                                                                       k=n_init)

    df = pd.DataFrame({
        't': t_selected,
        'p': p_selected,
        'cat': cat_selected,
        'arbr': arbr_selected,
        'pyrr': pyrr_selected
    })
    df['yields'] = df.apply(
        lambda row: objective_function(extraction_df, [row['t'], row['p'], row['cat'], row['arbr'], row['pyrr']]),
        axis=1)

    n_iter = 0
    while n_iter < n_total_iter:
        df = beta_acq(df, extraction_df)
        last_five = df.tail(5)
        for idx, row in last_five.iterrows():
            new_yield = objective_function(extraction_df, [row['t'], row['p'], row['cat'], row['arbr'], row['pyrr']])
            df.at[idx, 'yields'] = new_yield
        n_iter += 5

    done_yield_rows = df[df['yields'] > done_threshold]
    if not done_yield_rows.empty:
        max_yield_index = done_yield_rows.index.min()
        done_index = done_yield_rows.index.min()
        max_yield = df.loc[max_yield_index, 'yields']
    else:
        max_yield_index = df['yields'].idxmax()
        max_yield = df.loc[max_yield_index, 'yields']
        done_index = -1

    return pd.DataFrame({
        'seed': [seed],
        'max_yield_index': [max_yield_index],
        'done_index': [done_index],
        'max_yield': [max_yield]
    })


def evaluate_xgboost_params(xgboost_params, iteration):

    params_record = {
        'iteration': iteration,
        'n_estimators': int(xgboost_params['n_estimators']),
        'max_depth': int(xgboost_params['max_depth']),
        'learning_rate': xgboost_params['learning_rate'],
        'subsample': xgboost_params['subsample'],
        'colsample_bytree': xgboost_params['colsample_bytree'],
        'reg_alpha': xgboost_params['reg_alpha'],
        'reg_lambda': xgboost_params['reg_lambda']
    }


    extraction_df = train_xgboost_and_predict(xgboost_params)

    n_seed = 1000
    done_threshold = 0.80
    n_init = 15
    n_total_iter = 85

    seeds = range(2000, 2000 + n_seed)
    args_list = [(seed, n_init, n_total_iter, done_threshold, extraction_df) for seed in seeds]

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_seed)(args) for args in args_list
    )

    done_df = pd.concat(results).sort_values('seed').reset_index(drop=True)
    success_rate = (done_df['done_index'] != -1).mean()

    filename = f"rv_ofatbo.csv"
    done_df.to_csv(filename, index=False)


    return success_rate


def main():

    xgboost_params_iteration_2 = {
        'n_estimators': 109.0,
        'max_depth': 6.0,
        'learning_rate': 0.1514738597380149,
        'subsample': 0.8593292354322716,
        'colsample_bytree': 0.8186705157299192,
        'reg_alpha': 0.0971988081347264,
        'reg_lambda': 1.4552430554022893
    }

    success_rate = evaluate_xgboost_params(xgboost_params_iteration_2, 2)


if __name__ == '__main__':
    main()