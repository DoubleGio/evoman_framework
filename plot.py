## Creating the lineplots
## VU - Evolutionary Algorithms - team 49
## September 2021
from matplotlib import pyplot as plt
import numpy as np

N_GENS = 25
N_TESTS = 10

fig, axs = plt.subplots(2, 3, sharex='all', sharey='all', figsize=(14, 8))
for EA in range(0, 2):
    for ENEMY in range(0, 3):
        best = np.zeros((0, 25))
        means = np.zeros((0, 25))
        for i in range(N_TESTS):
            if EA == 1:
                file = np.loadtxt(f'./EA2 results/enemy{ENEMY+1}_test{i+1}/results.txt', skiprows=1, max_rows=N_GENS)
                best = np.vstack((best, file[:, 1]))
                means = np.vstack((means, file[:, 2]))
            else:
                file = np.loadtxt(f'./EA1 results/Pygad_enemy{ENEMY+1}_test{i+1}/results.txt', skiprows=1, max_rows=N_GENS)
                best = np.vstack((best, file[:, 3]))
                means = np.vstack((means, file[:, 1]))

        best_mean = np.mean(best, axis=0)
        best_std = np.std(best, axis=0)

        means_mean = np.mean(means, axis=0)
        means_std = np.std(means, axis=0)
        axs[EA, ENEMY].set_title(f'EA {EA+1} - Enemy {ENEMY+1}')
        axs[EA, ENEMY].plot(best_mean, label='mean of bests', color='tab:orange', marker='o')
        axs[EA, ENEMY].fill_between(range(25), best_mean-best_std, best_mean+best_std, color='tab:orange', alpha=0.5)

        axs[EA, ENEMY].plot(means_mean, label='mean of means', color='tab:blue', marker='o')
        axs[EA, ENEMY].fill_between(range(25), means_mean-means_std, means_mean+means_std, color='tab:blue', alpha=0.5)

for ax in axs.flat:
    ax.set(xlabel='Run #', ylabel='Fitness')
    ax.set_ylim(-10, 100)
for ax in axs.flat:
    ax.label_outer()

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='lower left')
# fig.tight_layout()
plt.subplots_adjust(left=0.055,
                    bottom=0.075,
                    right=0.98,
                    top=0.95,
                    wspace=0.06,
                    hspace=0.11)
plt.savefig('plots/lineplots.png')
plt.show()
