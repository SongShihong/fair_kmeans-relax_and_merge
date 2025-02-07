import numpy as np
from scipy.spatial.distance import cdist
# from cplex import Cplex
from gurobipy import *


def fair_partial_assignment(df, centers, alpha, beta, color_flag, rounding=False, type = GRB.CONTINUOUS):
    problem = Model('mip')
    
    clients = df.values
    n = clients.shape[0]
    k = centers.shape[0]
    dist = cdist(clients, centers)
    squared_dist = dist * dist
    
    # Add varibles
    # phi is the assignment matrix
    phi = problem.addVars(n, k, vtype = type, ub = 1, name = 'assignment')
    
    # Set cost function
    problem.setObjective(sum(phi[(i,j)]* squared_dist[i,j] for i in range(n) for j in range(k)), GRB.MINIMIZE)
    
    # Set sum to one constraints
    problem.addConstrs(sum(phi[(i, j)] for j in range(k)) == 1 for i in range(n))
    
    # Set fair constraints
    for var in color_flag:
        for color in range(min(color_flag[var]), max(color_flag[var])+1):
            # clients_belong_color is a vector with the same size with clients
            # if client_belong_color[i] = 1, that means the i th client belong to this color
            clients_belong_color = [i == color for i in color_flag[var]]
            problem.addConstrs((sum(clients_belong_color[i] * phi[(i, j)] for i in range(n)) <= alpha[var][color] * sum(phi[(i, j)] for i in range(n))) for j in range(k))
            problem.addConstrs((sum(clients_belong_color[i] * phi[(i, j)] for i in range(n)) >= beta[var][color] * sum(phi[(i, j)] for i in range(n))) for j in range(k))
     
    # close the output
    problem.setParam('outPutFlag',0)
    
    # optimize the model
    problem.optimize()
    
    res = {
            "status": problem.Status,
            "objective": problem.ObjVal,
            "assignment": np.array(list(problem.getAttr('x', phi).values())).reshape([n, k])
        }
    
    return res