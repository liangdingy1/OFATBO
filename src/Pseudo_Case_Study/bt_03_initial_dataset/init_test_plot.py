import pandas as pd
from matplotlib import pyplot as plt
import imageio.v2 as imageio


n_inits = range(5, 21)
n_seed = 1000
done_threshold = 99
n_total_iter = 100

results_df = pd.DataFrame(columns=['n_init', 'fail_count', 'success_rate'])

for n_init in n_inits:
    file_path = f'init_test_parallel_only/init_test_init{n_init}_seed{n_seed}_threshold{done_threshold}.csv'

    done_df = pd.read_csv(file_path)

    done_index_range = range(-1, 101)
    count_df = pd.DataFrame({'done_index': done_index_range})

    done_counts = done_df['done_index'].value_counts().reindex(done_index_range, fill_value=0)

    count_df['done_count_num'] = done_counts.values

    fail_count = done_counts[-1]
    success_rate = (1000 - fail_count) / 1000

    results_df = pd.concat(
        [results_df, pd.DataFrame({'n_init': [n_init], 'fail_count': [fail_count], 'success_rate': [success_rate]})],
        ignore_index=True)

    print(f"{n_init},{fail_count},{success_rate:.1%}")

    seed = done_df['seed'].values
    max_yield_index = done_df['max_yield_index'].values
    done_index = done_df['done_index'].values
    max_yield = done_df['max_yield'].values

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.scatter(seed, max_yield_index, color='blue', label='Max Yield Index', alpha=0.3)
    ax1.set_xlabel('Seed')
    ax1.set_ylabel('Index', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(-1, n_total_iter)
    ax1.scatter(seed, done_index, color='green', label='Done Index', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.scatter(seed, max_yield, color='red', label='Max Yield', alpha=0.7)
    ax2.set_ylabel('Max Yield', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(f'BETABO Stability Test (n_init={n_init})')

    plot1_path = f'init_test_parallel/init_test_init{n_init}_plot_1.png'
    plt.savefig(plot1_path)
    plt.close()

    plt.figure(figsize=(15, 6))
    plt.xlim(0, 100)
    plt.ylim(0, 70)
    plt.bar(count_df['done_index'], count_df['done_count_num'], color='blue', alpha=0.7, label='Done Count')
    plt.title(f'Done Index vs Done Count Num (n_init={n_init})')
    plt.xlabel('Done Index')
    plt.ylabel('Done Count Num')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plot2_path = f'init_test_parallel/plot_2_from0/init_test_init{n_init}_plot_2.png'
    plt.savefig(plot2_path)
    plt.close()

results_df.to_csv(f'init_test_parallel/init_test_success_rate.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(results_df['n_init'], results_df['success_rate'], marker='o', color='blue', linestyle='-', linewidth=2, markersize=8, label='Success Rate')

plt.title('Success Rate vs n_init')
plt.xlabel('n_init')
plt.ylabel('Success Rate')

plt.ylim(0.4, 0.8)

plt.grid(True, linestyle='--', alpha=0.5)

plt.legend()

plot3_path = f'init_test_parallel/init_test_success_rate.png'
plt.savefig(plot3_path)

plt.close()

image_files = [fr'init_test_parallel/plot_2_from0/init_test_init{n_init}_plot_2.png' for n_init in range(5, 21)]
images = [imageio.imread(file) for file in image_files]
imageio.mimsave(fr'init_test_parallel/plot_2_from0/init_test_plot_2_animation.gif', images, fps=1)