###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob, os

DOM_U = 1
DOM_L = -1
MUTATION = 0.2

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'EA1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
n_hidden_neurons = 10
# default environment fitness is assumed for experiment
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")
env.state_to_log()  # checks environment state
# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# genetic algorithm params
run_mode = 'train'  # train or test


def main():
    n_pop = 10
    n_gens = 5

    ####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
    ini = time.time()  # sets time marker

    # loads file with the best solution for testing
    if run_mode == 'test':
        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed', 'normal')
        evaluate([bsol])
        sys.exit(0)

    # initializes population loading old solutions or generating new ones

    print('\nNEW EVOLUTION\n')
    pop = np.random.uniform(DOM_L, DOM_U, (n_pop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = float(np.mean(fit_pop))
    std = float(np.std(fit_pop))
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

    # saves results for first pop
    file_aux = open(experiment_name + '/results.txt', 'a')
    file_aux.write('\n\ngen best mean std')
    print('\n GENERATION ' + str(0) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
        round(std, 6)))
    file_aux.write(
        '\n' + str(0) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.close()

    # evolution

    last_sol = fit_pop[best]
    notimproved = 0

    for gen_i in range(n_gens):

        offspring = crossover(pop, fit_pop)  # crossover
        fit_offspring = evaluate(offspring)  # evaluation
        pop = np.vstack((pop, offspring))  # new pop
        fit_pop = np.append(fit_pop, fit_offspring)  # new fitness results

        best = np.argmax(fit_pop)  # best solution in generation
        fit_pop[best] = float(evaluate(np.array([pop[best]]))[0])  # repeats best eval, for stability issues
        best_sol = fit_pop[best]

        # selection
        # avoiding negative probabilities, as fitness is ranges from negative numbers
        fit_pop_norm = minmax_norm(fit_pop)
        probs = fit_pop_norm / fit_pop_norm.sum()
        chosen = np.random.choice(pop.shape[0], n_pop, p=probs, replace=False)
        chosen = np.append(chosen[1:], best)
        pop = pop[chosen]
        fit_pop = fit_pop[chosen]

        # searching new areas
        if best_sol <= last_sol:
            notimproved += 1
        else:
            last_sol = best_sol
            notimproved = 0

        if notimproved >= 15:
            file_aux = open(experiment_name + '/results.txt', 'a')
            file_aux.write('\ndoomsday')
            file_aux.close()

            pop, fit_pop = doomsday(pop, fit_pop)
            notimproved = 0

        best = np.argmax(fit_pop)
        std = float(np.std(fit_pop))
        mean = float(np.mean(fit_pop))

        # saves results
        file_aux = open(experiment_name + '/results.txt', 'a')
        print('\n GENERATION ' + str(gen_i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' +
              str(round(std, 6)))
        file_aux.write('\n' + str(gen_i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' +
                       str(round(std, 6)))
        file_aux.close()

        # saves generation number
        file_aux = open(experiment_name + '/gen.txt', 'w')
        file_aux.write(str(gen_i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name + '/best.txt', pop[best])

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()

    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

    file = open(experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    env.state_to_log()  # checks environment state


# runs simulation
def simulation(x):
    f, _, _, _ = env.play(pcont=x)
    return f


# min-max normalization
def minmax_norm(arr):
    min_x = min(arr)
    max_x = max(arr)
    return np.array(list(map(lambda y: (y - min_x) / (max_x - min_x), arr)))


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(y), x)))


# tournament
def tournament(pop, fit_pop):
    n_options = 2 if pop.shape[0] < 3 else pop.shape[0] // 2  # N possible parents = 10% of pop or 2

    i_options = np.random.randint(0, pop.shape[0], n_options)  # get indices of random options
    fit_options = fit_pop.take(i_options)  # get fitness of these options
    options = np.vstack((i_options, fit_options))  # add to one matrix

    i_parents = np.flip(options[:, options[1].argsort()], 1)[0, 0:2]  # sort this matrix on fitness, select best two
    return pop[int(i_parents[0])], pop[int(i_parents[1])]  # return the two best parents


# limits
def limits(x):
    if x > DOM_U:
        return DOM_U
    elif x < DOM_L:
        return DOM_L
    else:
        return x


# crossover
def crossover(pop, fit_pop):
    total_offspring = np.zeros((0, n_vars))

    for p in range(0, pop.shape[0], 2):
        p1, p2 = tournament(pop, fit_pop)

        # n_offspring = np.random.randint(1, 4, 1)[0]
        n_offspring = 2
        offspring = np.zeros((n_offspring, n_vars))

        for f in range(0, n_offspring):

            cross_prop = np.random.uniform(0, 1)  # proportion of p1 that gets to cross over
            cross_mask = np.random.choice([True, False], size=n_vars, p=[cross_prop, 1 - cross_prop])
            offspring[f] = np.where(cross_mask, p1, p2)  # where mask = True, insert value from p1, else insert p2

            # mutation
            for i in range(0, len(offspring[f])):
                if np.random.uniform(0, 1) <= MUTATION:
                    # offspring[f][i] = offspring[f][i] + np.random.normal(0, 1)  # non-uniform mutation
                    offspring[f][i] = np.random.uniform(DOM_L, DOM_U)   # uniform mutation
            # offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f]))) # unnecessary for uniform mutation

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring


# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop, fit_pop):
    worst = int(pop.shape[0] / 4)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0, n_vars):
            pro = np.random.uniform(0, 1)
            if np.random.uniform(0, 1) <= pro:
                pop[o][j] = np.random.uniform(DOM_L, DOM_U)  # random dna, uniform dist.
            else:
                pop[o][j] = pop[order[-1:]][0][j]  # dna from best

        fit_pop[o] = evaluate([pop[o]])

    return pop, fit_pop


if __name__ == "__main__":
    main()
