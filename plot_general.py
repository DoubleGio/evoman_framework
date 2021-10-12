# Creating the lineplots
# VU - Evolutionary Algorithms - team 49
# September 2021
from matplotlib import pyplot as plt
import numpy as np

N_GENS = 25
N_TESTS = 10
FOLDERS = ['EA Pleuntje results/enemy[1, 3, 5, 7]', 'EA Daan results/enemy[2, 4, 6, 8]',
           'EA Chantal results/enemy[1, 3, 5, 7]', 'EAGio results/enemy[2, 4, 6, 8]']

TITLES = ['EA Det. - Odd', 'EA Det. - Even', 'EA Ada. - Odd', 'EA Ada. - Even']
fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
for i, EA in enumerate(TITLES):
    best = np.zeros((0, 25))
    means = np.zeros((0, 25))
    for j in range(N_TESTS):
        file = np.loadtxt(f'./{FOLDERS[i]}_test{j + 1}/results.txt', skiprows=1, max_rows=N_GENS)
        best = np.vstack((best, file[:, 1]))
        means = np.vstack((means, file[:, 2]))

    best_mean = np.mean(best, axis=0)
    best_std = np.std(best, axis=0)

    means_mean = np.mean(means, axis=0)
    means_std = np.std(means, axis=0)

    axs.flatten()[i].set_title(TITLES[i])
    axs.flatten()[i].plot(best_mean, label='mean of bests', color='tab:orange', marker='o')
    axs.flatten()[i].fill_between(range(25), best_mean - best_std, best_mean + best_std, color='tab:orange', alpha=0.5)

    axs.flatten()[i].plot(means_mean, label='mean of means', color='tab:blue', marker='o')
    axs.flatten()[i].fill_between(range(25), means_mean - means_std, means_mean + means_std, color='tab:blue', alpha=0.5)

for ax in axs.flat:
    ax.set(xlabel='Generation #', ylabel='Fitness')
    ax.set_ylim(0, 100)
for ax in axs.flat:
    ax.label_outer()

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='lower left')

plt.savefig(f'plots/general lineplots.png')
plt.show()
