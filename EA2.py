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

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'enemy1_test2_gio'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

dom_u = 1
dom_l = -1
npop = 40
gens = 25
mutation = 0.1
last_best = 0


# runs simulation
def simulation(env, x):
    f, _, _, _ = env.play(pcont=x)
    return f


# min-max normalization
def minmax_norm(arr):
    min_x = min(arr)
    max_x = max(arr)
    return np.array(list(map(lambda y: (y - min_x) / (max_x - min_x), arr)))


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


# tournament
def tournament(pop):
    n_options = 3 if pop.shape[0] < 40 else pop.shape[0] // 10  # N possible parents = 10% of pop or 2
    # n_options = 6
    i_options = np.random.randint(0, pop.shape[0], n_options)  # get indices of random options
    fit_options = fit_pop.take(i_options)  # get fitness of these options
    options = np.vstack((i_options, fit_options))  # add to one matrix

    i_parents = np.flip(options[:, options[1].argsort()], 1)[0, 0:3]  # sort this matrix on fitness, select best three
    return pop[int(i_parents[0])], pop[int(i_parents[1])], pop[int(i_parents[2])]  # return the three best parents


# limits
def limits(x):
    if x > dom_u:
        return dom_u
    elif x < dom_l:
        return dom_l
    else:
        return x


# crossover
def crossover(pop):
    total_offspring = np.zeros((0, n_vars))

    for p in range(0, pop.shape[0], 2):
        p1, p2, p3 = tournament(pop)
        n_offspring = 3
        offspring = np.zeros((n_offspring, n_vars))

        for f in range(0, n_offspring):

            # cross_prop = np.random.uniform(0,1)
            # offspring[f] = p1*cross_prop+p2*(1-cross_prop)
            cross_mask = np.random.choice([0, 1, 2], size=n_vars)
            offspring[f] = np.where(cross_mask == 0, p1, (np.where(cross_mask == 1, p2, p3)))
            # mutation
            for i in range(0, len(offspring[f])):
                if np.random.uniform(0, 1) <= mutation:
                    # offspring[f][i] = offspring[f][i]+np.random.normal(0, 1)
                    offspring[f][i] = np.random.uniform(dom_l, dom_u)

            # offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

        total_offspring = np.vstack((total_offspring, offspring))

    return total_offspring


# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop, fit_pop):
    worst = int(npop / 5)  # a fifth of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0, n_vars):
            pop[o][j] = np.random.uniform(dom_l, dom_u)  # random dna, uniform dist.
        fit_pop[o] = evaluate([pop[o]])

    return pop, fit_pop


# loads file with the best solution for testing
if run_mode == 'test':
    bsol = np.loadtxt(experiment_name + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    evaluate([bsol])

    sys.exit(0)

# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name + '/evoman_solstate'):

    print('\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print('\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux = open(experiment_name + '/gen.txt', 'r')
    ini_g = int(file_aux.readline())
    file_aux.close()

# saves results for first pop
file_aux = open(experiment_name + '/results.txt', 'a')
file_aux.write('\n\ngen best mean std')
print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
    round(std, 6)))
file_aux.write(
    '\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
file_aux.close()

# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g + 1, gens):

    offspring = crossover(pop)  # crossover
    fit_offspring = evaluate(offspring)  # evaluation

    order = np.argsort(fit_pop)
    deaths = order[0:int(npop * 0.8)]
    fit_pop = np.delete(fit_pop, deaths)
    pop = np.delete(pop, deaths, axis=0)

    pop = np.vstack((pop, offspring))
    fit_pop = np.append(fit_pop, fit_offspring)

    best = np.argmax(fit_pop)  # best solution in generation
    fit_pop[best] = float(evaluate(np.array([pop[best]]))[0])  # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    # selection
    # fit_pop_cp = fit_pop
    # fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
    # fit_pop_norm = minmax_norm(fit_pop)

    ranking = np.argsort(fit_pop)
    ranked_pop = pop[ranking]
    indices = np.arange(0, pop.shape[0])
    # s = 2
    # probs = ((2-s)/pop.shape[0]) + (2*indices*(s-1))/(pop.shape[0] * (pop.shape[0] - 1))
    # probs = (fit_pop_norm)/(fit_pop_norm).sum()
    p = (1 - np.exp(-indices))
    probs = p / np.sum(p)
    chosen = np.random.choice(pop.shape[0], npop, p=probs, replace=False)
    chosen = np.append(chosen[1:], best)
    pop = pop[chosen]
    fit_pop = fit_pop[chosen]

    # searching new areas

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 10:
        file_aux = open(experiment_name + '/results.txt', 'a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        pop, fit_pop = doomsday(pop, fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std = np.std(fit_pop)
    mean = np.mean(fit_pop)

    # saves results
    file_aux = open(experiment_name + '/results.txt', 'a')
    print('\n GENERATION ' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
        round(std, 6)))
    file_aux.write(
        '\n' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.close()

    # saves generation number
    file_aux = open(experiment_name + '/gen.txt', 'w')
    file_aux.write(str(i))
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

file = open(experiment_name + '/results.txt', 'a')
file.write('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')
file.close()

env.state_to_log()  # checks environment state
