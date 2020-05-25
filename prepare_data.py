"""
Helper functions to convert the data to the format expected by run_robot.py
"""

import sys
import seir
import pandas as pd
import numpy as np
import numpy.linalg as la
import os.path as path

# To use PyJulia
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main as Julia
Julia.eval('ENV["OMP_NUM_THREADS"] = 8')
Julia.eval('include("robot_dance.jl")')


def save_basic_parameters(tinc=5.2, tinf=2.9, rep=2.5, ndays=400, window=14, min_level=1.0, 
    hammer_level=0.89, hammer_durarion=0):
    """Save the basic_paramters.csv file using the data used in the report.

       All values are optional. If not present the values used in the report wihtout
       an initial hammer phase are used.
    """
    basic_prm = pd.Series(dtype=np.float)
    basic_prm["tinc"] = tinc
    basic_prm["tinf"] = tinf
    basic_prm["rep"] = rep
    basic_prm["ndays"] = ndays
    basic_prm["window"] = window
    basic_prm["min_level"] = min_level
    basic_prm["hammer_level"] = hammer_level
    basic_prm["hammer_duration"] = hammer_durarion
    basic_prm.to_csv(path.join("data", "basic_parameters.csv"), header=False)
    return basic_prm


def initial_conditions(basic_prm, city_data, min_days, Julia, correction=1.0):
    """Fits data and define initial contidions of the SEIR model.
    """
    population = city_data["estimated_population_2019"].iloc[0]
    confirmed = city_data["confirmed"]

    # Compute the new cases from the confirmed sum
    new_cases = confirmed.values[1:] - confirmed.values[:-1]

    # Use a mean in a week to smooth the data (specially to deal with weekends)
    observed_I = np.convolve(new_cases, np.ones(7, dtype=int), 'valid') / 7.0

    # Now accumulate in the inf_window
    inf_window = int(round(basic_prm["tinf"]))
    observed_I = np.convolve(observed_I, np.ones(inf_window, dtype=int), 'valid')

    ndays = len(observed_I)
    if ndays >= min_days:
        observed_I /= population
        Julia.observed_I = correction*observed_I
        Julia.tinc = basic_prm["tinc"]
        Julia.tinf = basic_prm["tinf"]
        Julia.rep = basic_prm["rep"]
        Julia.eval('initialc = fit_initial(tinc, tinf, rep, observed_I)')
        S0 = Julia.initialc[0]
        E0 = Julia.initialc[1]
        I0 = Julia.initialc[2]
        R0 = Julia.initialc[3]
        return (S0, E0, I0, R0, ndays), observed_I, population
    else:
        raise ValueError("Not enough data for %s only %d days available" % 
            (city_data["city"].iloc[0], len(observed_I)))


def simulate(parameters, city_data, min_days):
    """Simulate from the computed initial parameters until the last day.
    """
    c = city_data["city"].iloc[0]
    last_day = city_data["date"].iloc[-1]

    S0, E0, I0, R0, ndays = parameters[c]
    covid = seir.seir(ndays)
    print("Simulating", c, "until", last_day)
    result = covid.run((S0,E0,I0,R0))
    return result[:, -1], last_day


def compute_initial_condition_evolve_and_save(basic_prm, large_cities, min_pop, correction):
    """Compute the initial conditions and population and save it to data/cities_data.csv.

    The population andinitial condition is estimated  from a file with the information on
    the total number of confimed cases for the cities. See the example in
    data/covid_with_cities.csv.

    Parameters: large_cities: list with the name of cities tha are pre_selected.
        basic_prm: basic paramters for SEIR model
        large_cinties: minimal subset of cities do be selected.
        min_pop: minimal population to select more cities. 
        correction: a constant to multiply the observed cases to try to correct 
            subnotification. 
        TINF: Time of infection, should be equal to the tinf value in the 
            basic_parameters.csv file.
    """
    raw_epi_data = pd.read_csv("data/covid_with_cities.csv")
    raw_epi_data = raw_epi_data[raw_epi_data["state"] == "SP"]
    large_cities.extend(
        raw_epi_data[raw_epi_data["estimated_population_2019"] > min_pop]["city"].unique()
    )
    large_cities.sort()

    # Create a new Dataframe with only the needed information
    raw_epi_data = raw_epi_data[["city", "date", "confirmed", "estimated_population_2019"]]
    epi_data = raw_epi_data[raw_epi_data["city"] == large_cities[0]].copy()
    epi_data.sort_values(by=["date"], inplace=True)
    for city_name in large_cities[1:]:
        city = raw_epi_data[raw_epi_data["city"] == city_name].copy()
        city.sort_values(by = ["date"], inplace=True)
        epi_data = epi_data.append(city)
    epi_data.reset_index(inplace=True, drop=True)

    # Compute initial parameters fitting the data
    min_days = 5    
    parameters = {}
    ignored = []

    population = []
    n_cities = len(large_cities)
    for i in range(n_cities):
        city_name = large_cities[i]
        print("%d/%d" %(i + 1, n_cities), city_name)
        try:
            city_data = epi_data[epi_data["city"] == city_name]
            parameters[city_name], observed_I, city_pop = initial_conditions(basic_prm, 
                city_data, min_days, Julia, correction)
            population.append(city_pop)
        except ValueError:
            print("Ignoring ", city_name, "not enough data.")
            ignored.append(city_name)    

    # Simulate the data until the last day to start the optimization phase.
    cities_data = {}
    for city_name in large_cities:
        city_data = epi_data[epi_data["city"] == city_name]
        cities_data[city_name], last_day = simulate(parameters, city_data, min_days)

    # Save results
    cities_data = pd.DataFrame.from_dict(cities_data, 
        orient="index", columns=["S0", "E0", "I0", "R0"])
    cities_data["population"] = population
    with open('data/cities_data.csv', 'w') as f:
        f.write("# Initial condition for " + str(last_day) + "\n")
        cities_data.to_csv(f)
    return cities_data


def convert_mobility_matrix_and_save(cities_data, max_neighbors):
    """Read the mobile matrix data given by Pedro and save it in the format needed by
       robot_dance.

       cd: a data frame in the format of cities_data.csv
       max_neighbors: maximum number of neighbors allowed in the mobility matrix.
    """
    # Read the mobility_matrix
    large_cities = cities_data.index
    if path.exists("data/move_mat_SÃO PAULO_SP-Municipios_norm.csv"):
        mobility_matrix = pd.read_csv("data/move_mat_SÃO PAULO_SP-Municipios_norm.csv",
            header=None, sep=" ")
        cities_names = pd.read_csv("data/move_mat_SÃO PAULO_SP-Municipios_reg_names.txt", 
            header=None)

        # Cut the matrix to see only the desired cities
        cities_names = [i.title() for i in cities_names[0]]
        mobility_matrix.index = cities_names
        mobility_matrix.columns = cities_names
        mobility_matrix = mobility_matrix.loc[large_cities, large_cities].T
        mobility_matrix = mobility_matrix.mask(
            mobility_matrix.rank(axis=1, method='min', ascending=False) > max_neighbors + 1, 0
        )
    else:
        ncities = len(large_cities)
        pre_M = np.zeros((ncities, ncities))
        mobility_matrix = pd.DataFrame(data=pre_M, index=large_cities, columns=large_cities)

    # Adjust the mobility matrix
    np.fill_diagonal(mobility_matrix.values, 0.0)
    # out vector has at entry i the proportion of the population of city i that leaves the
    # city during the day
    out = mobility_matrix.sum(axis = 1)

    # The M matrix has at entry [i, j] the proportion, with respect to the population of j, 
    # of people from i that spend the day in j
    population = cities_data["population"]
    for i in mobility_matrix.index:
        mobility_matrix.loc[i] = (mobility_matrix.loc[i] * population[i] / 
                    population)
    mobility_matrix["out"] = out
    mobility_matrix.to_csv(path.join("data", "mobility_matrix.csv"))
    return mobility_matrix


def save_target(cities_data, target):
    """Save the target for maximum level of inffected.
    """
    large_cities = cities_data.index
    _ncities, ndays = target.shape
    days = list(range(1, ndays + 1))
    target_df = pd.DataFrame(data=target, index=cities_data.index, columns=days)
    target_df.to_csv(path.join("data", "target.csv"))
    return target_df
