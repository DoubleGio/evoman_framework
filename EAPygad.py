###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import pygad

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'Pygad9'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

run_mode = 'train' # train or test

if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    env.play(bsol)

    sys.exit(0)

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1

#first inputs, randomly chosen
function_inputs = np.random.uniform(dom_l, dom_u, n_vars)

#implement fitness function
def on_generation(ga):
    file_aux = open(experiment_name + '/results.txt', 'a')
    generation = ga.generations_completed
    print("Generation", generation)
    fitness = ga_instance.last_generation_fitness
    mean = str(np.mean(fitness))
    std = str(np.std(fitness))
    max = str(np.max(fitness))
    file_aux.write('\n' + str(generation) + ' ' + mean + ' ' + std + ' ' + max)
    print("MEAN = " + mean)
    print("STD = " + std)
    print("MAX = " + max)

def fitness_function(solution, solution_idx):
    output = solution*function_inputs
    fitness,p,e,t = env.play(output)
    return fitness

num_generations = 25
num_parents_mating = 4

sol_per_pop = 80
num_genes = len(function_inputs)

init_range_low = -2
init_range_high = 5

parent_selection_type = "sus"
keep_parents = 1

crossover_type = "scattered"

mutation_type = "swap"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=on_generation
                       )

ga_instance.run()



solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
np.savetxt(experiment_name+'/best.txt', solution)
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
prediction = np.sum(np.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))