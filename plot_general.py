## Creating the lineplots
## VU - Evolutionary Algorithms - team 49
## September 2021
from matplotlib import pyplot as plt
import numpy as np

N_GENS = 25
N_TESTS = 10
ENEMIES = [2, 4, 6, 8]
FOLDER = "EAGio results"

best = np.zeros((0, 25))
means = np.zeros((0, 25))
for i in range(N_TESTS):
    file = np.loadtxt(f'./{FOLDER}/enemy{ENEMIES}_test{i + 1}/results.txt', skiprows=1, max_rows=N_GENS)
    best = np.vstack((best, file[:, 1]))
    means = np.vstack((means, file[:, 2]))

best_mean = np.mean(best, axis=0)
best_std = np.std(best, axis=0)

means_mean = np.mean(means, axis=0)
means_std = np.std(means, axis=0)

plt.title(f'EA Adaptive - Enemies {ENEMIES}')
plt.xlabel('Generation #')
plt.ylabel('Fitness')
plt.ylim(-10, 100)
plt.plot(best_mean, label='mean of bests', color='tab:orange', marker='o')
plt.fill_between(range(N_GENS), best_mean - best_std, best_mean + best_std, color='tab:orange', alpha=0.5)

plt.plot(means_mean, label='mean of means', color='tab:blue', marker='o')
plt.fill_between(range(N_GENS), means_mean - means_std, means_mean + means_std, color='tab:blue', alpha=0.5)
plt.legend()

plt.savefig(f'plots/{FOLDER} lineplots.png')
plt.show()
