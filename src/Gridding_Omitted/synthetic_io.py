import imageio.v2 as imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import operation_window

n_init = 14
n_iter = 0
n_total_iter = 80
iter_count = 0


def f(T, t2, beta):
    f_T = 50 + 15 * np.exp(-((T + 5) ** 2) / 6) + 25 * np.exp(-((T - 5) ** 2) / 6)
    f_t2 = 40 + 30 * np.exp(-((t2 - 13) ** 2) / 6)
    f_beta = 70 + 20 * np.exp(-((beta - 115) ** 2) / 30)
    return (f_T * f_t2 * f_beta) / 4500


while n_iter < n_total_iter:

    operation_window.process_input_file(n_init, n_iter)

    output_df = pd.read_csv(fr'output_init{n_init}_iter{n_iter}.csv')

    last_five_indices = output_df.index[-5:]

    for idx in last_five_indices:
        T = output_df.at[idx, 'T']
        t2 = output_df.at[idx, 't2']
        beta = output_df.at[idx, 'beta']
        new_yield = f(T, t2, beta)
        output_df.at[idx, 'yield'] = new_yield

    n_iter += 5
    iter_count += 1
    output_df.to_csv(fr'input_init{n_init}_iter{n_iter}.csv', index=False)

    norm = Normalize(vmin=0, vmax=100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(fr'Iteration {iter_count}')

    scatter = ax.scatter(output_df['T'], output_df['t2'], output_df['beta'],
                         c=output_df['yield'], cmap='autumn_r', s=output_df['yield'], norm=norm)
    fig.colorbar(scatter)

    ax.set_xlabel('T')
    ax.set_ylabel('t2')
    ax.set_zlabel('beta')

    ax.view_init(elev=30, azim=45)

    ax.set_xlim(-20, 20)
    ax.set_ylim(5, 20)
    ax.set_zlim(100, 120)

    x_ticks = range(-20, 20 + 1, 10)
    y_ticks = range(5, 20 + 1, 5)
    z_ticks = range(100, 120 + 1, 5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)

    plt.savefig(fr'fig/iter{n_iter}.png', dpi=300)
    plt.show()

image_files = [fr'fig/iter{n_i}.png' for n_i in range(5, n_total_iter + 1, 5)]

max_images = 16
selected_files = image_files[:max_images]

images = [imageio.imread(file) for file in selected_files]

img_height, img_width = images[0].shape[:2]

grid_img = np.zeros((img_height * 4, img_width * 4, 3), dtype=np.uint8)

for idx, img in enumerate(images):
    row = idx // 4
    col = idx % 4
    if img.shape[2] == 4:
        img = img[:, :, :3]
    grid_img[row * img_height:(row + 1) * img_height,
    col * img_width:(col + 1) * img_width] = img

imageio.imwrite(fr'fig/fig16.png', grid_img)