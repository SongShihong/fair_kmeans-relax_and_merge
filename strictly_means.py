import configparser
import sys

from fair_clustering import strictly_fair_means
from util.configutil import read_list

from util.clusteringutil import subsample_data, read_data

import time
import tqdm

config_file = "config/main_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

config_str = "moons" if len(sys.argv) == 1 else sys.argv[1]

print("Using config_str = {}".format(config_str))

data_dir = config[config_str].get("data_dir")
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
k = list(map(int, config[config_str].getlist("num_clusters")))
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")
rounding = config["DEFAULT"].getboolean("rounding")
sample = config[config_str].getfloat('sample')

config.read(clustering_config_file)
df = read_data(config, dataset)

if max_points and len(df) > max_points:
    df = subsample_data(df, max_points)

for delta in deltas:
    print('delta = ', delta)
    for n_clusters in tqdm.tqdm(k):
        
        t1 = time.monotonic()
        # fair_clustering(dataset, df, clustering_config_file, n_clusters, delta, rounding, sample)
        cost, assignment = strictly_fair_means(dataset, df, clustering_config_file, n_clusters, delta)
        print('The integral is', cost)
        
        t2 = time.monotonic()
        print('the overall time', t2-t1)