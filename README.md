# Relax and Merge: A Simple Yet Effective Framework for Solving Fair k-Means and k-sparse Wasserstein Barycenter Problems

This code is the official implementation of the paper "Relax and Merge: A Simple Yet Effective Framework for Solving Fair k-Means and k-sparse Wasserstein Barycenter Problems" submitted to ICLR 2025.

## Requirements

Operating system: Ubuntu 20.04+

g++ and python3.10+ with optimizer `gurobi`, which requires a free educational license (apply from https://www.gurobi.com/).

To install requirements:

```setup
pip install -r requirements.txt
```

## Run an example

### Algorithm 1 ---- `main.py`

Run our main algorithm (Algorithm 1 in our paper) with default dataset (`bank`):

```
python main.py
```

If you want to change dataset, you can pass the argument of dataset like this:
```
python main.py dataset
```
The term should be replaced by the following options: `moons`, `bank`, `adult`, `creditcard`, `census`, `hypercube`, `cluto`, `balanced_cluto`, `complex`, `balanced_complex`.

### Algorithm 2 ---- `strictly_means.py`

Similar to `main.py`, our strictly fair k means algorithm (Algorithm 2 in our paper) with default dataset (`moons`) can be executed by the following command:

```
python strictly_means.py
```

You can also use other datset like this:
```
python strictly_means.py dataset
```

Note that `strictly_means.py` can only take strictly fair datasets, which are `moons`, `hypercube`, `cluto`, `balanced_cluto`, `complex` and `balanced_complex`.

## Advanced setting

The detailed information of datasets is stored in file `config/dataset_config.ini`.

The advanced setting of the main.py is stored in file `config/main_config.ini`.

If you want to add your own dataset, you need to add some configurations in the two above files following the existing records.

## Result

Take `moons` dataset as an example, after we run 
```
python main.py moons
```
or
```
python strictly_means.py moons
```

There may be some middle outputs, but the last lines should be like:
```
...
final LP of our method
 the cost is  99.08733479806166
the overall time 4.070996008813381
100%|███████| 1/1 [00:04<00:00,  4.07s/it]
```

## Refernces

The executive file `ApCentroid`, which is used to compute the epsilon-approximate centroid set, is based on the code of the paper "A local search approximation algorithm for k-means clustering" by Kanungo et al. Their open-source software is available in GNU programs: https://www.cs.umd.edu/users/mount/Projects/KMeans/.

We use some useful tools (filefolder `util/`) from the code of the paper "Fair Algorithms for Clustering" by Bera et al. to preprocess the data. Their source code has been opened in GitHub: https://github.com/nicolasjulioflores/fair_algorithms_for_clustering.