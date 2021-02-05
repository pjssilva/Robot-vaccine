"""
Helper functions to convert the data to the format expected by run_robot.py
"""

import sys
import pandas as pd
import numpy as np
import numpy.linalg as la
import os.path as path

import import_julia
import_julia.import_julia_and_robot_dance()
from julia import Main as Julia     # So we can call Julia variables using Julia.*

def save_basic_parameters(tinc=5.2, tinf=2.9, rep=2.5, ndays=400, time_icu=7, 
    alternate=1.0, window=14, min_level=1.0):
    """Save the basic_paramters.csv file using the data used in the report.

       All values are optional. If not present the values used in the report wihtout
       an initial hammer phase are used.
    """
    basic_prm = pd.Series(dtype=np.float)
    basic_prm["tinc"] = tinc
    basic_prm["tinf"] = tinf
    basic_prm["rep"] = rep
    basic_prm["ndays"] = ndays
    basic_prm["time_icu"] = time_icu
    basic_prm["alternate"] = alternate
    basic_prm["window"] = window
    basic_prm["min_level"] = min_level
    basic_prm.to_csv(path.join("data", "basic_parameters.csv"), header=False)
    return basic_prm


def convert_mobility_matrix_and_save(cities_data, max_neighbors, drs=None):
    """Read the mobility matrix data given by Pedro and save it in the format needed by
       robot_dance.

       cd: a data frame in the format of cities_data.csv
       max_neighbors: maximum number of neighbors allowed in the mobility matrix.
    """
    # Read the mobility_matrix
    large_cities = cities_data.index
    if drs is not None:
        mobility_matrix = pd.read_csv(drs, index_col=0)
        mobility_matrix = mobility_matrix.loc[large_cities, large_cities].T
        mobility_matrix = mobility_matrix.mask(
            mobility_matrix.rank(axis=1, method='min', ascending=False) > max_neighbors + 1, 0
        )
    elif path.exists("data/move_mat_SÃO PAULO_SP-Municipios_norm.csv"):
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
    #mobility_matrix *= 0.7
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


def save_hammer_data(cities_data, duration=0, level=0.89):
    """ Save hammer data
    """
    hammer_df = pd.DataFrame(index=cities_data.index)
    hammer_df["duration"] = duration
    hammer_df["level"] = level
    hammer_df.to_csv(path.join("data", "hammer_data.csv"))
    return hammer_df
