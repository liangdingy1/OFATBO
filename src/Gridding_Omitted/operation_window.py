import numpy as np
from argparse import Namespace
from dragonfly import load_config
from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.opt.gp_bandit import CPGPBandit
from dragonfly.exd.worker_manager import SyntheticWorkerManager
import pandas as pd


def process_input_file(n_init, n_iter):
    snar_data = pd.read_csv(rf'input_init{n_init}_iter{n_iter}.csv')
    n_data = len(snar_data)

    T = snar_data['T'].values
    t2 = snar_data['t2'].values
    beta = snar_data['beta'].values
    yields = snar_data['yield'].values

    X = []
    Y = []
    for i in range(n_data):
        x = [T[i], t2[i], beta[i]]
        y = [yields[i]]
        X.append(x)
        Y.append(y)


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



    min_stdev_T_t2 = np.full((len(domain_vars[0]['items']), len(domain_vars[1]['items'])), np.inf)
    min_stdev_T_beta = np.full((len(domain_vars[0]['items']), len(domain_vars[2]['items'])), np.inf)
    min_stdev_t2_beta = np.full((len(domain_vars[1]['items']), len(domain_vars[2]['items'])), np.inf)

    results = []

    for T_idx, T_val in enumerate(domain_vars[0]['items']):
        for t2_idx, t2_val in enumerate(domain_vars[1]['items']):
            for beta_idx, beta_val in enumerate(domain_vars[2]['items']):
                x_raw = [T_val, t2_val, beta_val]
                x_input = [opt.func_caller.get_processed_domain_point_from_raw(x_raw)]

                mu, stdev = opt.gp.eval(x_input, uncert_form='std')
                results.append([T_val, t2_val, beta_val, mu[0], stdev[0]])

                if stdev[0] < min_stdev_T_t2[T_idx, t2_idx]:
                    min_stdev_T_t2[T_idx, t2_idx] = stdev[0]

                if stdev[0] < min_stdev_T_beta[T_idx, beta_idx]:
                    min_stdev_T_beta[T_idx, beta_idx] = stdev[0]

                if stdev[0] < min_stdev_t2_beta[t2_idx, beta_idx]:
                    min_stdev_t2_beta[t2_idx, beta_idx] = stdev[0]

    results_df = pd.DataFrame(results, columns=['T', 't2', 'beta', 'mu', 'stdev'])


    min_stdev_T_t2_df = pd.DataFrame(min_stdev_T_t2, columns=domain_vars[1]['items'], index=domain_vars[0]['items'])
    min_stdev_T_beta_df = pd.DataFrame(min_stdev_T_beta, columns=domain_vars[2]['items'], index=domain_vars[0]['items'])
    min_stdev_t2_beta_df = pd.DataFrame(min_stdev_t2_beta, columns=domain_vars[2]['items'], index=domain_vars[1]['items'])


    max_stdev_T_t2_val = np.max(min_stdev_T_t2)
    max_stdev_T_t2_idx = np.unravel_index(np.argmax(min_stdev_T_t2), min_stdev_T_t2.shape)
    max_T = domain_vars[0]['items'][max_stdev_T_t2_idx[0]]
    max_t2 = domain_vars[1]['items'][max_stdev_T_t2_idx[1]]

    max_stdev_T_beta_val = np.max(min_stdev_T_beta)
    max_stdev_T_beta_idx = np.unravel_index(np.argmax(min_stdev_T_beta), min_stdev_T_beta.shape)
    max_T_beta = domain_vars[0]['items'][max_stdev_T_beta_idx[0]]
    max_beta = domain_vars[2]['items'][max_stdev_T_beta_idx[1]]

    max_stdev_t2_beta_val = np.max(min_stdev_t2_beta)
    max_stdev_t2_beta_idx = np.unravel_index(np.argmax(min_stdev_t2_beta), min_stdev_t2_beta.shape)
    max_t2_beta = domain_vars[1]['items'][max_stdev_t2_beta_idx[0]]
    max_beta_t2 = domain_vars[2]['items'][max_stdev_t2_beta_idx[1]]

    max_stdev_values = [max_stdev_T_t2_val, max_stdev_T_beta_val, max_stdev_t2_beta_val]
    max_stdev_index = np.argmax(max_stdev_values)

    if max_stdev_index == 0:
        max_stdev = max_stdev_T_t2_val
        plt_T = max_T
        plt_t2 = max_t2
        plt_beta = None
    elif max_stdev_index == 1:
        max_stdev = max_stdev_T_beta_val
        plt_T = max_T_beta
        plt_t2 = None
        plt_beta = max_beta
    else:
        max_stdev = max_stdev_t2_beta_val
        plt_T = None
        plt_t2 = max_t2_beta
        plt_beta = max_beta_t2


    filtered_results = results_df[
        ((results_df['T'] == plt_T) | (plt_T is None)) &
        ((results_df['t2'] == plt_t2) | (plt_t2 is None)) &
        ((results_df['beta'] == plt_beta) | (plt_beta is None))
    ]

    if not isinstance(filtered_results, pd.DataFrame):
        raise ValueError("filtered_results 必须是一个 DataFrame")
    num_rows = filtered_results.shape[0]
    step = num_rows // 5
    indices = [i for i in range(0, min(num_rows, 5) * step, step)]
    selected_rows = filtered_results.iloc[indices, :3]

    selected_rows['yield'] = 0

    combined_data = pd.concat([snar_data, selected_rows], ignore_index=True)

    combined_data.to_csv(rf'output_init{n_init}_iter{n_iter}.csv', index=False)