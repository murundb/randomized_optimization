# Optimization Problem 1: Four Peaks
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import six

# Workaround since mlrose is outdated
sys.modules["sklearn.externals.six"] = six
import mlrose

rng = np.random.default_rng(8)
N = [5, 20, 50, 100]
num_trials = 3

results = dict()

n_iter = 0


# Plotting
_, axes = plt.subplots(1, 4, figsize=(20, 5))

for n in N:

    result = dict()

    cost_function = mlrose.FourPeaks(t_pct=0.15)
    four_peaks_problem = mlrose.DiscreteOpt(length=n, fitness_fn=cost_function, maximize=True, max_val=2)

    rhc_timing_list = list()
    rhc_best_state_list = list()
    rhc_best_fitness_list = list()
    rhc_fitness_iteration_list = list()

    rhc_best = 0
    rhc_best_curve = None

    for _ in range(num_trials):

        # Randomized Hill Climbing
        start_time = time.time()
        rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(four_peaks_problem, max_attempts=10, max_iters=np.inf, restarts=5, init_state=None, curve=True)
        rhc_time = time.time() - start_time
        rhc_timing_list.append(rhc_time)
        rhc_best_state_list.append(rhc_best_state)
        rhc_best_fitness_list.append(rhc_best_fitness)
        rhc_fitness_iteration_list.append(rhc_fitness_curve.shape[0])

        if (rhc_best_fitness >= rhc_best):
            rhc_best = rhc_best_fitness
            rhc_best_curve = rhc_fitness_curve


    rhc_dict = {"rhc timing": np.mean(rhc_timing_list), "rhc best fitness": np.mean(rhc_best_fitness_list), 
    "rhc iteration": np.mean(rhc_fitness_iteration_list)}


    sa_timing_list = list()
    sa_best_state_list = list()
    sa_best_fitness_list = list()
    sa_fitness_iteration_list = list()
    sa_best_curve = None
    sa_best = 0

    for _ in range(num_trials):

        # Simulated annealing
        start_time = time.time()
        sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(four_peaks_problem, schedule=mlrose.GeomDecay(), max_attempts=10, max_iters=np.inf, init_state=None, curve=True)
        sa_time = time.time() - start_time
        sa_timing_list.append(sa_time)
        sa_best_state_list.append(sa_best_state)
        sa_best_fitness_list.append(sa_best_fitness)
        sa_fitness_iteration_list.append(sa_fitness_curve.shape[0])

        if (sa_best_fitness >= sa_best):
            sa_best = sa_best_fitness
            sa_best_curve = sa_fitness_curve

    sa_dict = {"sa timing": np.mean(sa_timing_list), "sa best fitness": np.mean(sa_best_fitness_list), 
    "sa iteration": np.mean(sa_fitness_iteration_list)}


    ga_timing_list = list()
    ga_best_state_list = list()
    ga_best_fitness_list = list()
    ga_fitness_iteration_list = list()

    ga_best = 0
    ga_best_curve = None

    for _ in range(num_trials):

        # A Genetic Algorithm
        start_time = time.time()
        ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(four_peaks_problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=np.inf, curve=True)
        ga_time = time.time() - start_time
        ga_timing_list.append(ga_time)
        ga_best_state_list.append(ga_best_state)
        ga_best_fitness_list.append(ga_best_fitness)
        ga_fitness_iteration_list.append(ga_fitness_curve.shape[0])

        if (ga_best_fitness >= ga_best):
            ga_best = ga_best_fitness
            ga_best_curve = ga_fitness_curve

    ga_dict = {"ga timing": np.mean(ga_timing_list), "ga best fitness": np.mean(ga_best_fitness_list), 
    "ga iteration": np.mean(ga_fitness_iteration_list)}

    mimic_timing_list = list()
    mimic_best_state_list = list()
    mimic_best_fitness_list = list()
    mimic_fitness_iteration_list = list()

    mimic_best = 0
    mimic_best_curve = None

    for _ in range(num_trials):
        # MIMIC
        start_time = time.time()
        mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(four_peaks_problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=np.inf, curve=True)
        mimic_time = time.time() - start_time
        mimic_timing_list.append(mimic_time)
        mimic_best_state_list.append(mimic_best_state)
        mimic_best_fitness_list.append(mimic_best_fitness)
        mimic_fitness_iteration_list.append(mimic_fitness_curve.shape[0])

        if (mimic_best_fitness >= mimic_best):
            mimic_best = mimic_best_fitness
            mimic_best_curve = mimic_fitness_curve

    mimic_dict = {"mimic timing": np.mean(mimic_timing_list), "ga best fitness": np.mean(ga_best_fitness_list), 
    "ga fitness iteration": np.mean(mimic_fitness_iteration_list)}

    axes[n_iter].grid()
    axes[n_iter].set_title("Fitness Curve for N = {}".format(n))
    axes[n_iter].set_xlabel("Maximum Number of Iterations")
    axes[n_iter].set_ylabel("Best Fitness")
    axes[n_iter].plot(rhc_best_curve, color="b", label="RHC")
    axes[n_iter].plot(sa_best_curve, color="r", label="SA")
    axes[n_iter].plot(ga_best_curve, color="g", label="GA")
    axes[n_iter].plot(mimic_best_curve, color="m", label="MIMIC")
    axes[n_iter].legend(loc="best")

    print("N: ", n)
    print("RHC Average Time: ", np.mean(rhc_timing_list))
    print("SA Average Time: ", np.mean(sa_timing_list))
    print("GA Average Time: ", np.mean(ga_timing_list))
    print("MIMIC Average Time: ", np.mean(mimic_timing_list))
    print("")
    print("RHC Best Fitness: ", np.mean(rhc_best_fitness_list))
    print("SA Best Fitness: ", np.mean(sa_best_fitness_list))
    print("GA Best Fitness: ", np.mean(ga_best_fitness_list))
    print("MIMIC Best Fitness: ", np.mean(mimic_best_fitness_list))
    print("")
    print("RHC Total Iteration: ", np.mean(rhc_fitness_iteration_list))
    print("SA Total Iteration: ", np.mean(sa_fitness_iteration_list))
    print("GA Total Iteration: ", np.mean(ga_fitness_iteration_list))
    print("MIMIC Total Iteration: ", np.mean(mimic_fitness_iteration_list))
    print("")

    n_iter += 1

plt.savefig("Fitness Curve.png")




