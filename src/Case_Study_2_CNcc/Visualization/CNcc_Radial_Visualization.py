import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# 设置全局字体为Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体

# 数据准备
df = pd.read_csv('bb_30_CNcc_beta_acq_Radial_Visualization.csv')
params = ['t', 'p', 'cat', 'arbr', 'pyrr']
df_norm = (df[params] - df[params].min()) / (df[params].max() - df[params].min())
angles = np.linspace(0, 2*np.pi, len(params), endpoint=False).tolist() + [0]

# 定义颜色区间（从高到低）
color_stops = np.linspace(1.0, 0.0, 11)  # 1.0, 0.9, ..., 0.0
color_rgb = [
    (255,0,37), (251,38,37), (246,89,38), (242,138,39),
    (238,177,40), (234,195,40), (205,192,41),
    (155,168,42), (87,126,43), (21,87,44)
]
colors = [(r/255, g/255, b/255) for r,g,b in color_rgb]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})

# 归一化产量并分配颜色
norm_yields = (df['yields'] - df['yields'].min()) / (df['yields'].max() - df['yields'].min())
for i in range(len(df)):
    values = df_norm.iloc[i].values.tolist() + df_norm.iloc[i].values.tolist()[:1]
    yield_val = 1 - norm_yields.iloc[i]  # 反转使高值对应亮色
    color_idx = min(int(yield_val * 10), 9)
    ax.plot(angles, values, '-', linewidth=1.5, color=colors[color_idx], alpha=0.6)

# 标记最佳和最差实验（同样使用区间颜色）
best_idx = df['yields'].argmax()
worst_idx = df['yields'].argmin()
for idx, style in [(best_idx, 'o-'), (worst_idx, 'o--')]:
    values = df_norm.iloc[idx].values.tolist() + df_norm.iloc[idx].values.tolist()[:1]
    yield_val = 1 - norm_yields.iloc[idx]
    color_idx = min(int(yield_val * 10), 9)
    line = ax.plot(angles, values, style, linewidth=3, color=colors[color_idx],
                  markersize=6 if style=='o-' else 6)
    # 仅对最佳实验添加填充（透明度提高到0.4）
    # if style == 'o-':
    #     ax.fill(angles, values, color=colors[color_idx], alpha=0.05)
    # elif style == 's--':
    #     ax.fill(angles, values, color=colors[color_idx], alpha=0.15)

# Add yield information text box (moved to upper center)
# textstr = f'Best yield: {df["yields"].iloc[best_idx]:.2f}%\nWorst yield: {df["yields"].iloc[worst_idx]:.2f}%'
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
# ax.text(0.9, 0.95, textstr, transform=ax.transAxes, fontsize=16,
#         verticalalignment='bottom', horizontalalignment='center', bbox=props)

# 美化图形
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
new_labels = ['t', 'Volt.', 'Catal.', 'ArBr', 'Pyrr']
ax.set_xticklabels(new_labels, fontsize=24, weight='bold')
ax.set_rlabel_position(0)  # 设置径向标签的位置
ax.tick_params(pad=20)  # 增加标签与轴的距离
ax.set_yticklabels([])
ax.grid(True, alpha=0.3)
# plt.title('Radial Analysis with Discrete Color Ranges', pad=30, fontsize=24, weight='bold')

plt.tight_layout()

# 保存为PNG（高清）
plt.savefig('radial_visualization.png', dpi=300, bbox_inches='tight')

# 保存为TIFF（无损压缩）
plt.savefig('radial_visualization.tif', dpi=300, bbox_inches='tight',
            format='tiff')

# eps
plt.savefig('radial_visualization.eps', dpi=300, bbox_inches='tight',
            format='eps')

plt.show()