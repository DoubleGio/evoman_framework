## EA2
## VU - Evolutionary Algorithms - team 49
## September 2021

import sys

sys.path.insert(0, 'evoman')  # Quite confusing but for whatever reason this line is needed to make the game run at all
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os

# Constants
LIM_U = 1
LIM_L = -1
MUTATION = 0.1
N_POP = 50
N_GENS = 15
N_NEURONS = 10

# Disable visuals
HEADLESS = True
if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def main():
    n_runs = 1
    run_mode = 'train'
    enemies = [2, 4, 6, 8]

    # Optional command line arguments
    if len(sys.argv) > 1:
        if len(sys.argv) == 4:
            try:
                n_runs = int(sys.argv[1])
                if sys.argv[2] in ['test', 'train']:
                    run_mode = sys.argv[2]
                enemies = list(sys.argv[3])
            except:
                print("USAGE: python EA2.py n_runs run_mode [1, 2, ...]")
                exit()
        else:
            print("USAGE: python EA2.py n_runs run_mode [1, 2, ...]")
            exit()

    folder = 'EAGio results'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for run in range(n_runs):
        print(f'\n=========== RUN {run + 1} / {n_runs} ===========\n')
        experiment_loc = f'{folder}/enemy{enemies}_test{run + 1}'
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
        # Default environment fitness is assumed for experiment
        env.state_to_log()  # Checks environment state

        if run_mode == 'train':
            train(env, experiment_loc, run)

        if run_mode == 'test':
            test_best(env, experiment_loc)


# Train the EA
def train(env, experiment_name, run):
    ini = time.time()  # Time marker, to keep track of the experiment duration

    # Evolutionary Algorithm start

    # We make use of the standard 'demo_controller.py'
    # Number of weights for neural network with a single hidden layer, necessary for 'demo_controller.py'
    n_vars = (env.get_num_sensors() + 1) * N_NEURONS + (N_NEURONS + 1) * 5

    # Initial population (random uniform)
    print(' INITIALIZING GENERATION\n')
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

    # Start evolving
    for gen_i in range(1, N_GENS):
        # Create children
        children = crossover(pop, fit_pop, n_vars, gen_i)
        fit_children = evaluate(env, children)

        # Have 80% of the parents, sorted on fitness, die off
        # order = np.argsort(fit_pop)
        # deaths = order[0:int(N_POP * 0.8)]
        # fit_pop = np.delete(fit_pop, deaths)
        # pop = np.delete(pop, deaths, axis=0)

        # All parents die, children are new pop
        pop = children
        fit_pop = fit_children

        # Survival using exponential ranked selection
        # Rank the pop and fit_pop
        ranking = np.argsort(fit_pop)
        ranked_pop = pop[ranking]
        ranked_fit_pop = fit_pop[ranking]
        # Get the probabilities based on rank
        indices = np.arange(0, pop.shape[0])
        p = (1 - np.exp(-indices))
        probabilities = p / np.sum(p)

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


# Parent selection
def parent_selection(pop, fit_pop):
    n_options = 3 if pop.shape[0] < 40 else pop.shape[0] // 10  # N possible parents = 10% of pop or 3
    i_options = np.random.randint(0, pop.shape[0], n_options)  # get indices of random options
    fit_options = fit_pop.take(i_options)  # get fitness of these options
    options = np.vstack((i_options, fit_options))  # add to one matrix

    i_parents = np.flip(options[:, options[1].argsort()], 1)[0, 0:3]  # sort this matrix on fitness, select best three
    return pop[int(i_parents[0])], pop[int(i_parents[1])], pop[int(i_parents[2])]  # return these three best parents


# Children creation
def crossover(pop, fit_pop, n_vars, gen_i):
    total_offspring = np.zeros((0, n_vars))

    for p in range(0, pop.shape[0], 2):  # Loop for npop / 2
        p1, p2, p3 = parent_selection(pop, fit_pop)
        n_offspring = 3
        offspring = np.zeros((n_offspring, n_vars))

        for f in range(0, n_offspring):
            cross_mask = np.random.choice([0, 1, 2], size=n_vars)
            offspring[f] = np.where(cross_mask == 0, p1, (np.where(cross_mask == 1, p2, p3)))
            # mutation
            for i in range(0, len(offspring[f])):
                if np.random.uniform(0, 1) <= MUTATION:
                    # offspring[f][i] = np.random.uniform(LIM_L, LIM_U)
                    s = 1 - 0.9 * (gen_i / N_GENS)
                    offspring[f][i] = limits(offspring[f][i] + np.random.normal(0, s))

        total_offspring = np.vstack((total_offspring, offspring))

    return total_offspring


# limits
def limits(x):
    if x > LIM_U:
        return LIM_U
    elif x < LIM_L:
        return LIM_L
    else:
        return x


if __name__ == "__main__":
    main()
