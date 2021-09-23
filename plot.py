from matplotlib import pyplot as plt
import numpy as np

ENEMY = 1
N_GENS = 25
N_TESTS = 5
best = np.zeros((0, 25))
means = np.zeros((0, 25))
for i in range(N_TESTS):
    file = np.loadtxt(f'../enemy{ENEMY}_test{i+1}_gio/results.txt', skiprows=3, max_rows=25)
    best = np.vstack((best, file[:, 1]))
    means = np.vstack((means, file[:, 2]))

best_mean = np.mean(best, axis=0)
best_std = np.std(best, axis=0)

means_mean = np.mean(means, axis=0)
means_std = np.std(means, axis=0)

plt.plot(best_mean, label='mean of bests', color='tab:orange', marker='o')
plt.fill_between(range(25), best_mean-best_std, best_mean+best_std, color='tab:orange', alpha=0.5)

plt.plot(means_mean, label='mean of means', color='tab:blue', marker='o')
plt.fill_between(range(25), means_mean-means_std, means_mean+means_std, color='tab:blue', alpha=0.5)

plt.xlabel('Run #')
plt.ylabel('Fitness')
plt.legend()
plt.show()
