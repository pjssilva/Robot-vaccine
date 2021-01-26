import os
import pandas as pd
import numpy as np
import datetime
import itertools
import matplotlib.pylab as plt
from matplotlib import rc
rc("text", usetex=True)
rc("font", family="serif")

from importlib import reload

import run_robot
import prepare_data
reload(prepare_data)
reload(run_robot)

# Hack so we can access Julia variables (for debugging)
from julia import Main as Julia

basic_prm = prepare_data.save_basic_parameters(min_level=0.8, rep=2.5, ndays=140)
subnot_factor = 11.6
cities_data = pd.read_csv("data/cities_data.csv", index_col=0)

# cities_data = prepare_data.compute_initial_condition_evolve_and_save(basic_prm, "SP", ["SP"], 10000000, subnot_factor, 1, "data/report_covid_with_drs_07_29.csv")
cities_data

# Create a target matrix (max infected level)
ncities, ndays = len(cities_data.index), int(basic_prm["ndays"])
target = 0.8*np.ones((ncities, ndays))
target = prepare_data.save_target(cities_data, target)

# Use a forcedif that releases the cities in the end
force_dif = np.ones((ncities, ndays))
cities_data

if basic_prm["time_icu"] == 11:
    # Time series adjusted considering the mean ICU time is 11 days
    ts_sp = np.array([0.0074335, 0.01523406, -0.00186355, 0.0, 1.67356018, -0.68192908, np.sqrt(0.00023883),
        0.007682840158843, 0.007536060983504])
    ts_notsp = np.array([0.00520255, 0.01532709, 0.00044498, 0.0, 1.75553282, -0.76360711, np.sqrt(3.567E-05),
        0.005426447471187, 0.005282217308748])
elif basic_prm["time_icu"] == 7:
    # Time series adjusted considering the mean ICU time is 7 days
    ts_sp = np.array([0.01099859, 0.02236023, 0.00370254, 0.0, 1.79119571, -0.80552926, np.sqrt(0.00034005),
        0.011644768910252, 0.011221496171591])
    ts_notsp = np.array([0.0076481, 0.0218084, 0.00367839, 0.0, 1.81361379, -0.82550856, np.sqrt(8.028E-05),
        0.007907216664912, 0.007721801045322])
else:
    raise NotImplementedError

# # Index of the cities that form the Metropolitan area of São Paulo
# MASP = np.array([7, 10, 15, 16, 17, 22]) - 1

MASP = 0
# Mean time in ICU
time_icu = 7


ts_drs = np.ones((len(cities_data), len(ts_notsp)))
ts_drs *= ts_notsp
ts_drs[MASP, :] = ts_sp
ts_drs = pd.DataFrame(data=ts_drs, index=cities_data.index, columns=[
    "rho_min", "rho_max", "intercept", "trend", "phi_1", "phi_2", "sigma_omega", "state0", "state_less_1"
])
ts_drs["confidence"] = 0.9
ts_drs["time_icu"] = time_icu
cities_data = pd.concat([cities_data, ts_drs], axis=1)
cities_data

pd.set_option("display.width", 120)

# Simple function to run a test and save results
def run_a_test(basic_prm, result_file, figure_file, cities_data, M, target, force_dif, pools=None, budget=0, tests_off=np.zeros(0, int), tau=3, test_efficacy=0.8, daily_tests=0, proportional_tests=False, verbosity=1):
    run_robot.prepare_optimization(basic_prm, cities_data, M, target, hammer_data, force_dif, pools, 
        verbosity=verbosity, test_budget=budget, tests_off=tests_off, tau=tau, test_efficacy=test_efficacy, 
        daily_tests=daily_tests, proportional_tests=proportional_tests)
    run_robot.optimize_and_show_results(basic_prm, figure_file, result_file, cities_data, target, verbosity=verbosity)
    

# Define mobility matrix.
M = prepare_data.convert_mobility_matrix_and_save(cities_data, max_neighbors=5, drs="data/report_drs_mobility.csv")
hammer_data = prepare_data.save_hammer_data(cities_data, 0, basic_prm["min_level"])
run_robot.find_feasible_hammer(basic_prm, cities_data, M, target, hammer_data, out_file=None, 
    incr_all=True, verbosity=1)

all_tau = [3]
all_test_efficacy = [0.8]
all_daily_tests = [30000] #[30000] , 50000]
all_budgets = [20000000] #[0, 5000000, 10000000, 20000000]

budget0_run = False
for tau, test_efficacy, daily_tests, budget in itertools.product(all_tau, all_test_efficacy, all_daily_tests, all_budgets):

    if budget == 0 and budget0_run:
        continue

    print("******************** Running tau =", tau, "test_effic =", test_efficacy, "daily tests =", daily_tests, "budget =", budget)
    
    # Case 1 Optimal tests
    basic_prm["alternate"] = 0.0
    tests_off = np.zeros(0, int)
    result_file =  f"results/optimal_tests_budget_{budget:d}_daily_{daily_tests:d}_tau_{tau:d}_test_ef_{test_efficacy:f}.csv"
    figure_file = f"results/optimal_tests_budget_{budget:d}_daily_{daily_tests:d}_tau_{tau:d}_test_ef_{test_efficacy:f}.png"
    run_a_test(basic_prm, result_file, figure_file, cities_data, M, target, force_dif, None, budget, tests_off, 
        tau, test_efficacy, daily_tests);
    budget0_run = True

    if budget == 0:
        continue
    
    plt.close("all")
