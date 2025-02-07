import configparser
import time
from collections import defaultdict
from functools import partial
import numpy as np
from gurobipy import *
import pandas as pd
# from cplex_fair_assignment_lp_solver import fair_partial_assignment
# from cplex_fair_assignment_lp_solver import iterative_rounding_lp
# import gurobi_fair_assignment_lp_solver
from gurobi_fair_assignment_lp_solver import fair_partial_assignment
# from cplex_violating_clustering_lp_solver import violating_lp_clustering
from util.clusteringutil import (clean_data, read_data, scale_data,
                                 subsample_data, take_by_key,
                                 vanilla_clustering, write_fairness_trial)
from util.configutil import read_list
from sklearn.cluster import KMeans
import os
# from iterative_rounding import iterative_rounding_lp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from rounding import *

def fair_clustering(dataset, df, config_file, num_clusters, deltas, rounding = False, sample = 1.0):
    
    fp_color_flag, fp_alpha, fp_beta, df = get_fair_parameters(dataset, df, config_file, deltas)
    # Compute the approximate centroid set
    df.to_csv('clean_data.csv', sep=' ', header=None, index=False)
    os.system('./ApCentroid -d {0} -df clean_data.csv'.format(df.shape[1]))

    cluster_centers = np.loadtxt('example.txt', dtype=float, delimiter=' ')
    # remove some impossible points
    upper = df.max().values
    lower = df.min().values
    cluster_centers = np.concatenate([cluster_centers, np.random.uniform(low=lower, high=upper, size=(199, df.shape[1]))])
    # cluster_centers = cluster_centers[(np.nonzero(((cluster_centers < upper) * (cluster_centers > lower)).min(1)))]
    rows = np.random.choice(cluster_centers.shape[0], size=int(max(cluster_centers.shape[0] * sample, num_clusters)), replace=False)
    cluster_centers = cluster_centers[rows]
    print(" the size of T is ", cluster_centers.shape[0])
    
    t1 = time.monotonic()
    res = fair_partial_assignment(df, cluster_centers, fp_alpha, fp_beta, fp_color_flag, False)
    t2 = time.monotonic()
    print('the lp time', t2 - t1)
    print(' the cost is ', res['objective'])
    
    assignment_matrix = np.array(res['assignment']).reshape([len(df), cluster_centers.shape[0]])
    center_weight = assignment_matrix.sum(0)
    center_weight = np.maximum(center_weight, np.zeros_like(center_weight))
    # print(center_weight)
    
    kmeans = KMeans(num_clusters)
    kmeans.fit(cluster_centers, sample_weight=center_weight)
    initial_score = np.sqrt(-kmeans.score(df))
    pred = kmeans.predict(df)
    final_center = kmeans.cluster_centers_
    
    print('final LP of our method')
    res = fair_partial_assignment(df, final_center, fp_alpha, fp_beta, fp_color_flag, rounding)
    print(' the cost is ', res['objective'])
    
    # print('test', compute_violation(res['assignment'], fp_color_flag, fp_alpha, fp_beta))
    # print('assign')
    # print(res['assignment'])
    # print(res['assignment'].shape)
    # Added because sometimes the LP for the next iteration solves so 
    # fast that `write_fairness_trial` cannot write to disk
    # time.sleep(1) 
    
    if(rounding == True):
        print('Start rounding')
        rounded_cost, rounded_assinment = Round(df, final_center, fp_color_flag, res['assignment'])
        print(' the cost after round is ', rounded_cost)
        
        # Compute the violation
        print(' the violation is ', compute_violation(rounded_assinment, fp_color_flag, fp_alpha, fp_beta))
        # largest_violation = 0
        # for var in fp_color_flag:
        #     for color in range(min(fp_color_flag[var]), max(fp_color_flag[var])+1):
        #         clients_belong_color = [i == color for i in fp_color_flag[var]]
        #         clients_belong_color_index = np.nonzero(clients_belong_color)[0]
        #         color_rounded_phi = rounded_assinment[clients_belong_color_index]
        #         rounded_gamma = color_rounded_phi.sum(0)
        #         overall_gamma = rounded_assinment.sum(0)
        #         ratio = rounded_gamma / overall_gamma
                
        #         # Compute the violation
        #         alpha_violation = np.maximum((ratio - fp_alpha[var][color]), 0)
        #         beta_violation = np.maximum((fp_alpha[var][color] - ratio), 0)
        #         violation = np.maximum(alpha_violation, beta_violation).max()
        #         largest_violation = max(violation, largest_violation)
        # print(' the violation is ', largest_violation)

def compute_violation(rounded_assignment, fp_color_flag, fp_alpha, fp_beta):
    largest_violation = 0
    for var in fp_color_flag:
        for color in range(min(fp_color_flag[var]), max(fp_color_flag[var])+1):
            clients_belong_color = [i == color for i in fp_color_flag[var]]
            clients_belong_color_index = np.nonzero(clients_belong_color)[0]
            color_rounded_phi = rounded_assignment[clients_belong_color_index]
            rounded_gamma = color_rounded_phi.sum(0)
            overall_gamma = rounded_assignment.sum(0)
            ratio = rounded_gamma / overall_gamma
            
            # Compute the violation
            # alpha_violation = np.maximum((ratio - fp_alpha[var][color]), 0)
            # beta_violation = np.maximum((fp_beta[var][color] - ratio), 0)
            alpha_violation = np.maximum((rounded_gamma - fp_alpha[var][color] * overall_gamma), 0)
            beta_violation = np.maximum((fp_beta[var][color] * overall_gamma - rounded_gamma), 0)
            violation = np.maximum(alpha_violation, beta_violation).max()
            largest_violation = max(violation, largest_violation)
    return largest_violation

def vanilla_kmeans(dataset, df, config_file, num_clusters, deltas, rounding = False):
    fp_color_flag, fp_alpha, fp_beta, df = get_fair_parameters(dataset, df, config_file, deltas)
    initial_score, pred, cluster_centers = vanilla_clustering(df, num_clusters, 'kmeans')
    print(' the cost before rounding is ',initial_score)
    phi = np.zeros([df.shape[0], num_clusters])
    for i in range(df.shape[0]):
        phi[i, pred[i]] = 1
    print(' the violation is ', compute_violation(phi, fp_color_flag, fp_alpha, fp_beta))
    
def get_fair_parameters(dataset, df, config_file, delta):
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # df = read_data(config, dataset)
    df, _ = clean_data(df, config, dataset)
    


    # variable_of_interest (list[str]) : variables that we would like to collect statistics for
    variable_of_interest = config[dataset].getlist("variable_of_interest")

    # Assign each data point to a color, based on config file
    # attributes (dict[str -> defaultdict[int -> list[int]]]) : holds indices of points for each color class
    # color_flag (dict[str -> list[int]]) : holds map from point to color class it belongs to (reverse of `attributes`)
    attributes, color_flag = {}, {}
    for variable in variable_of_interest:
        colors = defaultdict(list)
        this_color_flag = [0] * len(df)
        
        condition_str = variable + "_conditions"
        bucket_conditions = config[dataset].getlist(condition_str)

        # For each row, if the row passes the bucket condition, 
        # then the row is added to that color class
        for i, row in df.iterrows():
            for bucket_idx, bucket in enumerate(bucket_conditions):
                if eval(bucket)(row[variable]):
                    colors[bucket_idx].append(i)
                    this_color_flag[i] = bucket_idx

        attributes[variable] = colors
        color_flag[variable] = this_color_flag

    # representation (dict[str -> dict[int -> float]]) : representation of each color compared to the whole dataset
    representation = {}
    for var, bucket_dict in attributes.items():
        representation[var] = {k : (len(bucket_dict[k]) / len(df)) for k in bucket_dict.keys()}

    # Select only the desired columns
    selected_columns = config[dataset].getlist("columns")
    df = df[[col for col in selected_columns]]
    
    
    
    print('the size of clean data is ', df.shape)
    # df.to_csv('clean_data.csv', sep=' ', header=None, index=False)
    
    # Compute the approximate centroid set
    # os.system('./ApCentroid -d {0} -df clean_data.csv'.format(df.shape[1]))

    # Scale data if desired
    scaling = config["DEFAULT"].getboolean("scaling")
    if scaling:
        df = scale_data(df)

    # fairness_vars (list[str]) : Variables to perform fairness balancing on
    fairness_vars = config[dataset].getlist("fairness_variable")
        #   alpha_i = a_val * (representation of color i in dataset)
        #   beta_i  = b_val * (representation of color i in dataset)
    num_color = 0
    alpha, beta = {}, {}
    # for delta in deltas:
    a_val, b_val = 1 / (1 - delta), 1 - delta
    for var, bucket_dict in attributes.items():
        alpha[var] = {k : a_val * representation[var][k] for k in bucket_dict.keys()}
        beta[var] = {k : b_val * representation[var][k] for k in bucket_dict.keys()}
        num_color = num_color + len(alpha[var])
    
    print('the number of colors is', num_color)
    
    color_flag = take_by_key(color_flag, fairness_vars)
    alpha = take_by_key(alpha, fairness_vars)
    beta = take_by_key(beta, fairness_vars)
    # fp_color_flag, fp_alpha, fp_beta 
    return (color_flag, alpha, beta, df)

# def base_masc(dataset, df, config_file, num_clusters, deltas):

def baseline_ijcai(dataset, df, config_file, num_clusters, deltas, rounding = False):
    color_flag, alpha, beta, df = get_fair_parameters(dataset, df, config_file, deltas)
    
    best_res = None
    best_val = float('inf')
    
    for var in color_flag:
        for color in range(min(color_flag[var]), max(color_flag[var])+1):
            # clients_belong_color is a vector with the same size with clients
            # if client_belong_color[i] = 1, that means the i th client belong to this color
            clients_belong_color = [i == color for i in color_flag[var]]
            clients_belong_color_points = df.values[np.nonzero(clients_belong_color)[0]]
            
            # in each color, we compute the k-means as the location
            kmeans = KMeans(min(num_clusters, clients_belong_color_points.shape[0]))
            kmeans.fit(clients_belong_color_points)
            cluster_centers = kmeans.cluster_centers_
            res = fair_partial_assignment(df, cluster_centers, alpha, beta, color_flag)
            
            # choose the best one
            if res['objective'] < best_val:
                best_val = res['objective']
                best_res = res
    
    return best_res

def baseline_orl(dataset, df, config_file, num_clusters, deltas):
    color_flag, alpha, beta, df = get_fair_parameters(dataset, df, config_file, deltas)

    best_means_cost = float('inf')
    best_assignment = None
    num_color = 0
    
    for var in color_flag:
        for color in range(min(color_flag[var]), max(color_flag[var])+1):
            num_color = num_color + 1
            labels = np.zeros(df.shape[0])
            # clients_belong_color is a vector with the same size with clients
            # if client_belong_color[i] = 1, that means the i th client belong to this color
            clients_belong_color = [i == color for i in color_flag[var]]
            clients_belong_color_points = df.values[np.nonzero(clients_belong_color)[0]]
            pivital_points = clients_belong_color_points
            
            kmeans = KMeans(min(num_clusters, pivital_points.shape[0]))
            kmeans.fit(clients_belong_color_points)
            cluster_centers = kmeans.cluster_centers_
            pivital_points_assignment = kmeans.predict(clients_belong_color_points)
            
            data_center_dist = cdist(df.values, cluster_centers)
            data_center_cost = data_center_dist * data_center_dist
            
            means_cost = 0
            for another_var in color_flag:
                for another_color in range(min(color_flag[another_var]), max(color_flag[another_var])+1):
                    another_clients_belong_color = [i == another_color for i in color_flag[another_var]]
                    another_clients_belong_color_index = np.nonzero(another_clients_belong_color)[0]
                    another_clients_belong_color_points = df.values[another_clients_belong_color_index]
                    dist = cdist(clients_belong_color_points, another_clients_belong_color_points)
                    squared_dist = dist * dist
                    row_ind, col_ind = linear_sum_assignment(squared_dist)
                    
                    for i in range(pivital_points.shape[0]):
                        corres_data_index = another_clients_belong_color_index[col_ind]
                        label = pivital_points_assignment[i]
                        means_cost += (data_center_cost[corres_data_index[i], label])
                        labels[corres_data_index] = label
            
            if means_cost < best_means_cost:
                best_means_cost = means_cost
                best_assignment = labels
    
    return best_means_cost, best_assignment

def refine_strict(dataset, df, config_file, num_clusters, deltas):
    color_flag, alpha, beta, df = get_fair_parameters(dataset, df, config_file, deltas)

    best_means_cost = float('inf')
    best_assignment = None
    
    for var in color_flag:
        for color in range(min(color_flag[var]), max(color_flag[var])+1):
            labels = np.zeros(df.shape[0])
            # clients_belong_color is a vector with the same size with clients
            # if client_belong_color[i] = 1, that means the i th client belong to this color
            clients_belong_color = [i == color for i in color_flag[var]]
            clients_belong_color_points = df.values[np.nonzero(clients_belong_color)[0]]
            pivital_points = clients_belong_color_points
            
            centroids = pivital_points
            fd = {}
            num_color = 0
            for another_var in color_flag:
                for another_color in range(min(color_flag[another_var]), max(color_flag[another_var])+1):
                    num_color += 1
                    another_clients_belong_color = [i == another_color for i in color_flag[another_var]]
                    another_clients_belong_color_index = np.nonzero(another_clients_belong_color)[0]
                    another_clients_belong_color_points = df.values[another_clients_belong_color_index]
                    dist = cdist(clients_belong_color_points, another_clients_belong_color_points)
                    squared_dist = dist * dist
                    row_ind, col_ind = linear_sum_assignment(squared_dist)
                    fd[(another_var, another_color)] = col_ind  
                    
                    centroids = centroids + another_clients_belong_color_points[col_ind]
            
            centroids = centroids / num_color
            kmeans = KMeans(min(num_clusters, centroids.shape[0]))
            kmeans.fit(centroids)
            cluster_centers = kmeans.cluster_centers_
            centroid_assignment = kmeans.predict(centroids)
            data_center_dist = cdist(df.values, cluster_centers)
            data_center_cost = data_center_dist * data_center_dist
            
            means_cost = 0
            for another_var in color_flag:
                for another_color in range(min(color_flag[another_var]), max(color_flag[another_var])+1):
                    another_clients_belong_color = [i == another_color for i in color_flag[another_var]]
                    another_clients_belong_color_index = np.nonzero(another_clients_belong_color)[0]
                    another_clients_belong_color_points = df.values[another_clients_belong_color_index]
                    for i in range(pivital_points.shape[0]):
                        corres_data_index = another_clients_belong_color_index[col_ind]
                        label = centroid_assignment[i]
                        means_cost += (data_center_cost[corres_data_index[i], label])
                        labels[corres_data_index] = label
            if means_cost < best_means_cost:
                best_means_cost = means_cost
                best_assignment = labels
    
    return best_means_cost, best_assignment

def strictly_fair_means(dataset, df, config_file, num_clusters, deltas):
    color_flag, alpha, beta, df = get_fair_parameters(dataset, df, config_file, deltas) 
    
    best_sumation = None
    best_var = None
    best_color = None
    best_fd_cost = float('inf')
    best_fd = None
    num_color = 0
    
    for var in color_flag:
        for color in range(min(color_flag[var]), max(color_flag[var])+1):
            num_color = num_color + 1
            # clients_belong_color is a vector with the same size with clients
            # if client_belong_color[i] = 1, that means the i th client belong to this color
            clients_belong_color = [i == color for i in color_flag[var]]
            clients_belong_color_points = df.values[np.nonzero(clients_belong_color)[0]]
            
            fd_cost = 0
            fd = {}
            sumation = clients_belong_color_points
            for another_var in color_flag:
                for another_color in range(min(color_flag[another_var]), max(color_flag[another_var])+1):
                    another_clients_belong_color = [i == another_color for i in color_flag[another_var]]
                    another_clients_belong_color_points = df.values[np.nonzero(another_clients_belong_color)[0]]
                    dist = cdist(clients_belong_color_points, another_clients_belong_color_points)
                    squared_dist = dist * dist
                    row_ind, col_ind = linear_sum_assignment(squared_dist)
                    for r, c in zip(row_ind, col_ind):
                        fd_cost += squared_dist[r, c]
                    fd[(another_var, another_color)] = col_ind
                    sumation = sumation + another_clients_belong_color_points[col_ind]
            
            if fd_cost < best_fd_cost:
                best_fd_cost = fd_cost
                best_sumation = sumation
                best_var = var
                best_color = color
                best_fd = fd
                
    centroids = best_sumation / num_color
    kmeans = KMeans(min(num_clusters, centroids.shape[0]))
    kmeans.fit(centroids)
    cluster_centers = kmeans.cluster_centers_
    centroids_assignment = kmeans.predict(centroids)
    
    overall_cost = 0
    labels = np.zeros(df.shape[0])
    pivital_color = [i == best_color for i in color_flag[best_var]]
    pivital_points = df.values[np.nonzero(pivital_color)[0]]
    data_center_dist = cdist(df.values, cluster_centers)
    data_center_cost = data_center_dist * data_center_dist
    for var in color_flag:
        for color in range(min(color_flag[var]), max(color_flag[var]) + 1):
            clients_belong_color = [i == color for i in color_flag[var]]
            clients_belong_color_index = np.nonzero(clients_belong_color)[0]
            clients_belong_color_points = df.values[clients_belong_color_index]
            
            # for i in range(pivital_points.shape[0]):
            #     corres_data_index = clients_belong_color_index[best_fd[(var, color)][i]]
            #     label = centroids_assignment[i]
            #     overall_cost += (data_center_cost[corres_data_index, label]).sum()
            #     labels[corres_data_index] = label
            for i in range(pivital_points.shape[0]):
                corres_data_index = clients_belong_color_index[best_fd[(var, color)][i]]
                label = centroids_assignment[i]
                overall_cost += (data_center_cost[corres_data_index, label]).sum()
                labels[corres_data_index] = label
    
    return overall_cost, labels