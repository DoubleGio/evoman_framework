## EA1
## VU - Evolutionary Algorithms - team 49
## September 2021

import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
import pygad

# Constants
LIM_U = 1
LIM_L = -1
N_POP = 80
N_GENS = 25
N_NEURONS = 10

# Disable visuals
HEADLESS = True
if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def main():
    n_runs = 10
    run_mode = 'train'
    enemy = 3

    # Optional command line arguments
    if len(sys.argv) > 1:
        if len(sys.argv) == 4:
            try:
                n_runs = int(sys.argv[1])
                if sys.argv[2] in ['test', 'train']:
                    run_mode = sys.argv[2]
                enemy = int(sys.argv[3])
            except:
                print("USAGE: python EA1.py n_runs run_mode enemy")
                exit()
        else:
            print("USAGE: python EA1.py n_runs run_mode enemy")
            exit()

    if not os.path.exists('EA1 results'):
        os.makedirs('EA1 results')

    for run in range(n_runs):
        print(f'\n=========== RUN {run + 1} / {n_runs} ===========\n')
        global experiment_loc
        experiment_loc = f'EA1 results/enemy{enemy}_test{run + 1}'
        if not os.path.exists(experiment_loc):
            os.makedirs(experiment_loc)

        global env
        env = Environment(experiment_name=experiment_loc,
                          enemies=[enemy],
                          playermode="ai",
                          player_controller=player_controller(N_NEURONS),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          randomini="yes")
        # Default environment fitness is assumed for experiment
        env.state_to_log()  # Checks environment state

        if run_mode == 'train':
            train()

        if run_mode == 'test':
            test_best()


def train():
    ini = time.time()  # Time marker, to keep track of the experiment duration

    # Evolutionary Algorithm start

    # We make use of the standard 'demo_controller.py'
    # Number of weights for neural network with a single hidden layer, necessary for 'demo_controller.py'
    n_vars = (env.get_num_sensors() + 1) * N_NEURONS + (N_NEURONS + 1) * 5

    # Initializing inputs
    # function_inputs = np.zeros(n_vars)

    # Parameters
    num_parents_mating = 8
    parent_selection_type = "rws"
    keep_parents = 4

    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 15
    ga_instance = pygad.GA(num_generations=N_GENS,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=N_POP,
                           num_genes=n_vars,
                           init_range_low=LIM_L,
                           init_range_high=LIM_U,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           on_generation=on_generation,
                           save_best_solutions=True)

    ga_instance.run()

    generation = ga_instance.best_solution_generation
    solution = ga_instance.best_solutions[generation]
    solution_fitness = ga_instance.best_solutions_fitness[generation]
    print("Parameters of the best solution : {solution}".format(solution=solution))
    np.savetxt(experiment_loc + '/best.txt', solution)
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))


# Test the best found solution
def test_best():
    best_solution = np.loadtxt(experiment_loc + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    env.play(best_solution)


# Print the results after each generation
def on_generation(ga):
    file_aux = open(experiment_loc + '/results.txt', 'a')
    generation = ga.generations_completed
    print("Generation", generation)
    fitness = ga.last_generation_fitness
    mean = str(np.mean(fitness))
    std = str(np.std(fitness))
    best = str(np.max(fitness))
    file_aux.write('\n' + str(generation) + ' ' + mean + ' ' + std + ' ' + best)
    print("MEAN = " + mean)
    print("STD = " + std)
    print("MAX = " + best)


def fitness_function(solution, solution_idx):
    fitness, _, _, _ = env.play(solution)
    return fitness


if __name__ == "__main__":
    main()
