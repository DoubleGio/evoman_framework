# Creating the table
# VU - Evolutionary Algorithms - team 49
# September 2021
import sys

sys.path.insert(0, 'evoman')  # Quite confusing but for whatever reason this line is needed to make the game run at all
from environment import Environment
from demo_controller import player_controller
import os
from matplotlib import pyplot as plt
import numpy as np


N_RUNS = 10
N_REPEATS = 5
ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]
N_NEURONS = 10
FOLDERS = ['EA Pleuntje results/enemy[1, 3, 5, 7]', 'EA Daan results/enemy[2, 4, 6, 8]',
           'EA Chantal results/enemy[1, 3, 5, 7]', 'EAGio results/enemy[2, 4, 6, 8]']

# Disable visuals
HEADLESS = True
if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def custom_cons_multi(values):
    return values.mean()


best = []
data = {'EA Det. - Odd': [], 'EA Det. - Even': [], 'EA Ada. - Odd': [], 'EA Ada. - Even': []}
for i, key in enumerate(data):
    best = np.zeros((3, 10))
    for run in range(N_RUNS):
        experiment_loc = f'{FOLDERS[i]}_test{run + 1}'
        cur = np.zeros((3, 10))
        for repeat in range(N_REPEATS):
            e = np.zeros((3, 10))
            for j, enemy in enumerate(ENEMIES):
                env = Environment(experiment_name=experiment_loc,
                                  enemies=[enemy],
                                  playermode="ai",
                                  player_controller=player_controller(N_NEURONS),
                                  enemymode="static",
                                  level=2,
                                  speed="fastest",
                                  randomini="yes")
                best_sol = np.loadtxt(experiment_loc + '/best.txt')
                print(f'\n{FOLDERS[i]} - RUN {run + 1} - REPEAT {repeat + 1}\n')
                e[0, j], e[1, j], e[2, j], _ = env.play(pcont=best_sol)
            cur = cur + e
        avg_cur = cur / N_REPEATS
        if a[0].sum() > best[].sum()
        r.append(np.mean(f))
    data[key] = r

fig, ax = plt.subplots()
medianprops = dict(color='black')
box = ax.boxplot(data.values(), patch_artist=True, labels=data.keys(), medianprops=medianprops)
colours = ['royalblue', 'lightsteelblue', 'sandybrown', 'peachpuff']
for patch, colour in zip(box['boxes'], colours):
    patch.set_facecolor(colour)

ax.set_title(f'5 Tests vs. All Enemies')
ax.set_ylabel('Fitness')
ax.set_ylim(0, 100)

plt.savefig(f'plots/boxplots.png')
plt.show()
