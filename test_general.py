# Testing the solutions and creating the gain boxplots and table
# VU - Evolutionary Algorithms - team 49
# September 2021
import sys

sys.path.insert(0, 'evoman')  # Quite confusing but for whatever reason this line is needed to make the game run at all
from environment import Environment
from demo_controller import player_controller
import os
from matplotlib import pyplot as plt
from tabulate import tabulate
from scipy import stats
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


def main():
    best_sol = np.zeros(265)
    best_gain = -800  # -800 is the worst possible performance (800 is the best)
    best_results = np.zeros((10, 2))
    data = {'EA Det. - Odd': [], 'EA Det. - Even': [], 'EA Ada. - Odd': [], 'EA Ada. - Even': []}
    for i, key in enumerate(data):
        gains = np.zeros(10)
        for run in range(N_RUNS):
            experiment_loc = f'{FOLDERS[i]}_test{run + 1}'
            run_best_sol = np.loadtxt(experiment_loc + '/best.txt')
            res_run = np.zeros((len(ENEMIES), 2))  # Columns: Player life, Enemy life
            for repeat in range(N_REPEATS):
                print(f'\n{FOLDERS[i]} - RUN {run + 1} - REPEAT {repeat + 1}\n')
                env = Environment(experiment_name=experiment_loc,
                                  enemies=ENEMIES,
                                  playermode="ai",
                                  player_controller=player_controller(N_NEURONS),
                                  enemymode="static",
                                  level=2,
                                  speed="fastest",
                                  randomini="yes",
                                  multiplemode="yes")
                env.cons_multi = custom_cons_multi
                _, player_life, enemy_life, _ = env.play(pcont=run_best_sol)
                res_run[:, 0] += player_life
                res_run[:, 1] += enemy_life
            res_run = res_run / N_REPEATS  # Average results of all enemies for 1 run

            gains[run] = np.sum(res_run[:, 0]) - np.sum(res_run[:, 1])
            if gains[run] > best_gain:
                best_gain = gains[run]
                best_sol = run_best_sol
                best_results = res_run
        data[key] = gains
    np.savetxt('best_solution.text', best_sol)
    plot(data)
    ttest_odd = stats.ttest_ind(data['EA Det. - Odd'], data['EA Ada. - Odd'])
    ttest_even = stats.ttest_ind(data['EA Det. - Even'], data['EA Ada. - Even'])
    with open('table.txt', "w") as file:
        table = tabulate(best_results, showindex=ENEMIES, headers=['Enemy #', 'Player life', 'Enemy life'],
                         tablefmt='presto')
        print("\n BEST RESULTS: \n")
        print(table)
        print(f'\n BEST GAIN: {best_gain}')
        print(f'\n TTest Det. vs Ada. algorithm - ODD enemies, p-value = {ttest_odd.pvalue}')
        print(f'\n TTest Det. vs Ada. algorithm - EVEN enemies, p-value = {ttest_even.pvalue}')
        file.write(table)
        file.write(f'\n TTest Det. vs Ada. algorithm - ODD enemies, p-value = {ttest_odd.pvalue}')
        file.write(f'\n TTest Det. vs Ada. algorithm - EVEN enemies, p-value = {ttest_even.pvalue}')


def plot(data):
    fig, ax = plt.subplots()
    median_props = dict(color='black')
    box = ax.boxplot(data.values(), patch_artist=True, labels=data.keys(), medianprops=median_props)
    colours = ['royalblue', 'lightsteelblue', 'sandybrown', 'peachpuff']
    for patch, colour in zip(box['boxes'], colours):
        patch.set_facecolor(colour)

    ax.set_title(f'5 Tests vs. All Enemies')
    ax.set_ylabel('Gain')

    plt.savefig(f'plots/boxplots.png')
    plt.show()


def custom_cons_multi(values):
    return values


if __name__ == "__main__":
    main()
