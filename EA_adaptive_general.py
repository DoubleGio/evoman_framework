# EA2
# VU - Evolutionary Algorithms - team 49
# September 2021

import sys

sys.path.insert(0, 'evoman')  # Quite confusing but for whatever reason this line is needed to make the game run at all
from environment import Environment
from demo_controller import player_controller
from matplotlib import pyplot as plt
import time
import numpy as np
import os

# Constants
LIM_U = 1
LIM_L = -1
MUTATION = 0.1
N_POP = 80
N_GENS = 25
N_NEURONS = 10
FOLDER = 'EAGio results'

# Disable visuals
HEADLESS = True
if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def main():
    n_runs = 10
    run_mode = 'train'
    enemies = [2, 4, 6, 8]
    # enemies = [1, 3, 5, 7]

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    for run in range(n_runs):
        print(f'\n=========== RUN {run + 1} / {n_runs} ===========\n')
        experiment_loc = f'{FOLDER}/enemy{enemies}_test{run + 1}'
        if not os.path.exists(experiment_loc):
            os.makedirs(experiment_loc)

        env = Environment(experiment_name=experiment_loc,
                          enemies=enemies,
                          playermode="ai",
                          player_controller=player_controller(N_NEURONS),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          randomini="yes",
                          multiplemode="yes")
        env.cons_multi = custom_cons_multi
        # Default environment fitness is assumed for experiment
        env.state_to_log()  # Checks environment state

        if run_mode == 'train':
            train(env, experiment_loc, run)

        if run_mode == 'test':
            test_best(env, experiment_loc)
    # plot(n_runs, enemies)


# Train the EA
def train(env, experiment_name, run):
    ini = time.time()  # Time marker, to keep track of the experiment duration

    # Evolutionary Algorithm start

    # We make use of the standard 'demo_controller.py'
    # Number of weights for neural network with a single hidden layer, necessary for 'demo_controller.py'
    n_vars = (env.get_num_sensors() + 1) * N_NEURONS + (N_NEURONS + 1) * 5

    # Initial population (random uniform)
    print('\n   INITIALIZATION\n')
    pop = np.random.uniform(LIM_L, LIM_U, (N_POP, n_vars))
    fit_pop = evaluate(env, pop)
    best_i = np.argmax(fit_pop)
    mean = float(np.mean(fit_pop))
    std = float(np.std(fit_pop))
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

    # Write/log results for pop 0
    file_aux = open(f'{experiment_name}/results.txt', 'a')
    file_aux.write('gen best mean std')
    print(f'\n RUN {run + 1}: GENERATION 0 => '
          f'{str(round(fit_pop[best_i], 6))} {str(round(mean, 6))} {str(round(std, 6))}')
    file_aux.write(f'\n0 {str(round(fit_pop[best_i], 6))} {str(round(mean, 6))} {str(round(std, 6))}')
    file_aux.close()

    c = 0.9
    sigma = 1
    # Start evolving
    for gen_i in range(1, N_GENS):
        # Create children
        children = crossover(pop, fit_pop, n_vars, sigma)
        fit_children = evaluate(env, children)

        # 1/5 rule thingy
        if gen_i % 2 == 0:
            best = fit_pop.max()
            x = np.where(fit_children > best, True, False)
            percentage = np.sum(x) / fit_children.shape[0]
            if percentage > 0.2:
                sigma = sigma / c
            elif percentage < 0.2:
                sigma = sigma * c

        # Add children to the total pop
        pop = np.copy(children)
        fit_pop = np.copy(fit_children)

        # Survival using exponential ranked selection
        # Rank the pop and fit_pop
        ranking = np.argsort(fit_pop)
        ranked_pop = pop[ranking]
        ranked_fit_pop = fit_pop[ranking]
        # Get the probabilities based on rank
        indices = np.arange(0, pop.shape[0])
        s = 1.75
        probabilities = ((2 - s) / pop.shape[0]) + ((2 * (indices * (s - 1))) / (pop.shape[0] * (pop.shape[0] - 1)))

        # Choose the survivors + always add the highest ranked (can be added twice)
        chosen = np.random.choice(pop.shape[0], N_POP - 1, p=probabilities, replace=False)
        chosen = np.append(chosen, indices.max(initial=0))
        pop = ranked_pop[chosen]
        fit_pop = ranked_fit_pop[chosen]

        best_i = np.argmax(fit_pop)  # best solution in generation
        std = float(np.std(fit_pop))
        mean = float(np.mean(fit_pop))

        # saves results
        file_aux = open(f'{experiment_name}/results.txt', 'a')
        print(f'\n RUN {run + 1}: GENERATION {gen_i} => {str(round(fit_pop[best_i], 6))} '
              f'{str(round(mean, 6))} {str(round(std, 6))}')
        file_aux.write(f'\n{gen_i} {str(round(fit_pop[best_i], 6))} {str(round(mean, 6))} {str(round(std, 6))}')
        file_aux.close()

        # saves file with the best solution
        np.savetxt(f'{experiment_name}/best.txt', pop[best_i])

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()

    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

    file = open(f'{experiment_name}/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    file = open(f'{experiment_name}/results.txt', 'a')
    file.write('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')
    file.close()

    env.state_to_log()  # checks environment state


# Test the best found solution
def test_best(env, experiment_name):
    best_solution = np.loadtxt(experiment_name + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    evaluate(env, [best_solution])


# Takes a population, runs a simulation for each agent and returns an array containing the resulting fitnesses
def evaluate(env, x):
    return np.apply_along_axis(lambda y: env.play(pcont=y), 1, x)[:, 0]


# Tournament selection
def parent_selection(pop, fit_pop, sigma, n_parents=3):
    parents = np.zeros((n_parents, pop.shape[1]))
    percentage = (10 + (1 - sigma) * 40) / 100
    tournament_size = max(5, np.floor(pop.shape[0] * percentage))  # Tournament size = max(5, 10-50% of pop)
    for i in range(0, n_parents):
        i_options = np.random.randint(0, pop.shape[0], int(tournament_size))  # randomly pick some individuals
        fit_options = fit_pop.take(i_options)  # get fitness of these options
        options = np.vstack((i_options, fit_options))  # combine into one matrix
        parents[i] = pop[int(options[0, options[1].argmax()])]  # select the best option as parent
    return parents


# Children creation
def crossover(pop, fit_pop, n_vars, sigma=1):
    total_offspring = np.zeros((0, n_vars))

    for p in range(0, pop.shape[0], 2):  # Loop for npop / 2
        p1, p2, p3 = parent_selection(pop, fit_pop, sigma)
        n_offspring = 3
        offspring = np.zeros((n_offspring, n_vars))

        for f in range(0, n_offspring):
            cross_mask = np.random.choice([0, 1, 2], size=n_vars)
            offspring[f] = np.where(cross_mask == 0, p1, (np.where(cross_mask == 1, p2, p3)))
            # mutation
            for i in range(0, len(offspring[f])):
                if np.random.uniform(0, 1) <= MUTATION:
                    # offspring[f][i] = np.random.uniform(LIM_L, LIM_U)
                    offspring[f][i] = offspring[f][i] + np.random.normal(0, sigma)
            offspring[f] = np.clip(offspring[f], LIM_L, LIM_U)  # Values lower than -1 become -1, higher than 1 become 1

        total_offspring = np.vstack((total_offspring, offspring))

    return total_offspring


def custom_cons_multi(values):
    return values.mean()


def plot(n_tests, enemies):
    best = np.zeros((0, N_GENS))
    means = np.zeros((0, N_GENS))
    for i in range(n_tests):
        file = np.loadtxt(f'{FOLDER}/enemy{enemies}_test{i + 1}/results.txt', skiprows=1, max_rows=N_GENS)
        best = np.vstack((best, file[:, 1]))
        means = np.vstack((means, file[:, 2]))
    best_mean = np.mean(best, axis=0)
    best_std = np.std(best, axis=0)

    means_mean = np.mean(means, axis=0)
    means_std = np.std(means, axis=0)

    plt.title(f'EA Adaptive - Enemies {enemies}')
    plt.xlabel('Run #')
    plt.ylabel('Fitness')
    plt.ylim(-10, 100)
    plt.plot(best_mean, label='mean of bests', color='tab:orange', marker='o')
    plt.fill_between(range(N_GENS), best_mean - best_std, best_mean + best_std, color='tab:orange', alpha=0.5)

    plt.plot(means_mean, label='mean of means', color='tab:blue', marker='o')
    plt.fill_between(range(N_GENS), means_mean - means_std, means_mean + means_std, color='tab:blue', alpha=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
