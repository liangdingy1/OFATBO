import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='dragonfly.utils.oper_utils')
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


done_df = pd.read_csv(fr'function_2_bo.csv')


done_index_range = range(-1, 51)
count_df = pd.DataFrame({'done_index': done_index_range})

done_counts = done_df['done_index'].value_counts().reindex(done_index_range, fill_value=0)

count_df['done_count_num'] = done_counts.values

print(done_counts[-1])


seed = done_df['seed'].values
max_yield_index = done_df['max_yield_index'].values
done_index = done_df['done_index'].values
max_yield = done_df['max_yield'].values

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.scatter(seed, max_yield_index, color='blue', label='Max Yield Index', alpha=0.3)
ax1.set_xlabel('Seed')
ax1.set_ylabel('Index', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-1, 50)

ax1.scatter(seed, done_index, color='green', label='Done Index', alpha=0.3)

ax2 = ax1.twinx()

ax2.scatter(seed, max_yield, color='red', label='Max Yield', alpha=0.7)
ax2.set_ylabel('Max Yield', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 100)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.title('Bayesian Optimization Stability Test')

plot1_path = f'function_2_bo_test_plot_1.png'
plt.savefig(plot1_path)
plt.show()
plt.close()


plt.figure(figsize=(15, 6))
plt.bar(count_df['done_index'], count_df['done_count_num'], color='blue', alpha=0.7, label='Done Count')

plt.title('Done Index vs Done Count Num (Bar Chart)')
plt.xlabel('Done Index')
plt.ylabel('Done Count Num')

plt.grid(True, linestyle='--', alpha=0.5)

plt.legend()

plt.text(0.1, 0.1, f'failure_counts:{done_counts[-1]}\nsuccess rate:{((1000-done_counts[-1])/1000):.1%}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='bottom', horizontalalignment='left')

plot2_path = f'function_2_bo_test_plot_2.png'
plt.savefig(plot2_path)
plt.show()
plt.close()