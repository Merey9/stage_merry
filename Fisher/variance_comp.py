import os
import camb
import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
from tqdm import tqdm
import pandas as pd
from functions.simu import *
from functions.fisher_matrix import *

wanted_keys = ["TT", "EE", "BB", "TE", "EB", "TB"]

name_simus = "cov_test"

data = {name: [] for name in wanted_keys}
len_data = {name: 0 for name in wanted_keys}

for XY in wanted_keys:
    # Import the data
    data[XY] = pd.read_csv("Simu/cls_" + XY + "_" + name_simus + ".csv", header=None)
    len_data[XY] = data[XY].shape[1]
    assert data[wanted_keys[0]].shape[1] == data[XY].shape[1]

ls = np.arange(max(len_data.values()))
nu = 2 * ls + 1

assert len(ls) == data[wanted_keys[0]].shape[1]
assert len(nu) == data[wanted_keys[0]].shape[1]

# Calculating the Cls mean of the simulation
cls_mean = {}
for name in data.keys():
    cls_mean[name] = data[name].mean().to_numpy()

cov_matrix_mean = get_cov_matrix(cls_mean, fsky=1)
# print(cov_matrix_mean)
# Calculating variance in the simulation and theo variance from mean Cl
for i, XY in enumerate(wanted_keys):
    for j in range(i, len(wanted_keys)):
        WZ = wanted_keys[j]
        var_mean = cov_matrix_mean[:, i, j]
        var_eff = np.sum(
            (data[XY] - cls_mean[XY]) * (data[WZ] - cls_mean[WZ]), axis=0
        ).to_numpy() / (len(data[XY]) - 1)
        plt.plot(ls[2:], var_mean[2:], color="black")
        plt.plot(ls[2:], var_eff[2:], color="red", alpha=0.7)
        plt.semilogy()
        plt.savefig("Figures/Var_comp/var_comp_" + XY + WZ + "_" + name_simus + ".png")
        plt.clf()
