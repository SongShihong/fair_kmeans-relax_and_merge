import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
# from cplex import Cplex
from gurobipy import *

def Round(df, centers, color_flag, phi):
    
    data_center_dist = cdist(df.values, centers)
    data_center_cost = data_center_dist * data_center_dist
    
    rounded_cost = 0
    
    for var in color_flag:
        for color in range(min(color_flag[var]), max(color_flag[var])+1):
            # labels = np.zeros(df.shape[0])
            # clients_belong_color is a vector with the same size with clients
            # if client_belong_color[i] = 1, that means the i th client belong to this color
            clients_belong_color = [i == color for i in color_flag[var]]
            clients_belong_color_index = np.nonzero(clients_belong_color)[0]
            
            color_phi = phi[clients_belong_color_index]
            color_cost_matrix = data_center_cost[clients_belong_color_index]
            
            res = rounding_color(color_cost_matrix, color_phi)
            rounded_color_phi = res['assignment']
            rounded_cost += res['objective']
            phi[clients_belong_color_index] = rounded_color_phi
            
    return rounded_cost, phi

def rounding_color(color_cost_matrix, color_phi):
    
    # Gamma is the weight vector of centers
    gamma = color_phi.sum(0).reshape([-1, 1])
    floor_gamma = np.floor(gamma)
    fractional_gamma = gamma - floor_gamma
    
    ratio_floor = floor_gamma / gamma
    ratio_floor_phi = ratio_floor @ np.ones([ratio_floor.shape[1], color_phi.shape[1]])
    
    # The transmission matrix from center to clients
    # reversed_phi = color_phi.T
    reversed_cost = color_cost_matrix.T
    k, n = reversed_cost.shape
    
    # floor_phi = ratio_floor_phi * reversed_phi
    # fractional_phi = reversed_phi - floor_phi
    
    problem = Model('mip')
    integral_x = problem.addVars(k, n, vtype = GRB.BINARY, ub = 1)
    fractional_x = problem.addVars(k, n, vtype = GRB.BINARY, ub = 1)
    
    problem.setObjective(sum(integral_x[(j, i)]* reversed_cost[j, i] for i in range(n) for j in range(k)) + sum(fractional_x[(j, i)]* reversed_cost[j, i] for i in range(n) for j in range(k)), GRB.MINIMIZE)
    
    problem.addConstrs(sum(integral_x[(j, i)] for i in range(n)) == floor_gamma[j] for j in range(k))
    problem.addConstrs(sum(fractional_x[(j, i)] for i in range(n)) <= 1 for j in range(k))
    problem.addConstrs(sum(integral_x[(j, i)] + fractional_x[(j, i)] for j in range(k)) == 1 for i in range(n))
    
    # close the output
    problem.setParam('outPutFlag',0)
    
    # optimize the model
    problem.optimize()
    
    res = {
            "status": problem.Status,
            "objective": problem.ObjVal,
            'assignment': (np.array(list(problem.getAttr('x', integral_x).values())).reshape(k, n) + np.array(list(problem.getAttr('x', fractional_x).values())).reshape(k, n)).T
        }
    
    return res
    