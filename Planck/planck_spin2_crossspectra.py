import sys 
sys.path.append('/data/stage_merry/functions')
import healpy as hp
from pspy import *
from matplotlib import pyplot as plt
import numpy as np
from math import pi
import pysm3
from pysm3 import units as u
from tqdm import tqdm
from plot_functions import *
from fisher_matrix import *
from itertools import combinations_with_replacement, product

def fac(ls):
    return ls * (ls + 1) / (2 * pi)

spec_keys_pspy = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

comp_bool = True

date_float = "0527_0501"

cross_spec_key = 'TT'

filename = "Planck/Figures/Planck_all_band_%s_%s%s.png" % (cross_spec_key, date_float, "_comp" if comp_bool else "")

spec_xlims = {  'TT': (0, 3000),
                'EE': (0, 1500),
                'TE': (0, 1200),
                'ET': (0, 1200),
                'EB': (0, 1200),
                'BB': (0, 1500)}

spec_ylims = {  'TT': (0, 7000),
                'EE': (0, 70),
                'TE': (-200, 200),
                'ET': (-200, 200),
                'EB': (-10, 10),
                'BB': (-2, 10)}

spec_ylims_res = {  'TT': None,
                'TE': None,
                'EE': None,
                'ET': (-200, 200),
                'EB': (-10, 10),
                'BB': None}


BAND_1 = "100"
BAND_2 = "143"

BAND_iter = [("100", "100"), ("143", "143"), ("217", "217"), ("100", "143"), ("100", "217"), ("143", "217")]
# BAND_iter = [("100", "143")]
rows, cols = best_subplot_shape(len(BAND_iter))

fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*6), dpi=300)
axs = np.array(axs).reshape(rows, cols)
fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust the spacing between subplots
plt.tight_layout()

for i, (BAND_1, BAND_2) in tqdm(enumerate(BAND_iter)):
    row = i // cols
    col = i % cols
    axs[row, col].set_title("%sx%s" % (BAND_1, BAND_2))
    
    lb, Dls_12 = so_spectra.read_ps("data/spectra/maison/Dls_%sx%s_%s.dat" % (BAND_1, BAND_2, date_float), spectra=spec_keys_pspy)
    ls_21, Dls_21 = so_spectra.read_ps("data/spectra/maison/Dls_%sx%s_%s.dat" % (BAND_2, BAND_1, date_float), spectra=spec_keys_pspy)

    try:
        data_Planck = np.loadtxt("data/spectra/planck/planck_spectrum_%s_%sx%s.dat" % (cross_spec_key, BAND_1, BAND_2)).T
    except:
        data_Planck = "erm what the sigma"

    assert lb.all() == ls_21.all()

    Dls_sum = {cross_spec_key: np.zeros_like(Dls_12[cross_spec_key], dtype=float)}

    for key in [cross_spec_key, cross_spec_key[::-1]]:
        Dls_sum[cross_spec_key] += Dls_12[key]
        Dls_sum[cross_spec_key] += Dls_21[key]

    Dls_sum[cross_spec_key] /= 4

    if comp_bool:
        try:
            comp_indices = [i for i, value in enumerate(lb) if value in set(data_Planck[0])]
            Dls_comp = [Dls_sum[cross_spec_key][i] for i in comp_indices]
            Dls_res = Dls_comp - (data_Planck[1][:len(Dls_comp)] * fac(data_Planck[0][:len(Dls_comp)])) # / (data_Planck[key][2] * fac(data_Planck[key][0])
            axs[row, col].plot(data_Planck[0][:len(Dls_comp)], Dls_res, color='black', label="mean")
            axs[row, col].set_ylim(spec_ylims_res[cross_spec_key])
        except:
            pass
        
    else:
        for key in [cross_spec_key, cross_spec_key[::-1]]:
            axs[row, col].plot(lb, Dls_12[key], alpha=0.5, label="%s_%sx%s" % (key, BAND_1, BAND_2), linewidth=1)
            if BAND_1 != BAND_2:
                axs[row, col].plot(lb, Dls_21[key], alpha=0.5, label="%s_%sx%s" % (key, BAND_2, BAND_1), linewidth=1)
        axs[row, col].plot(lb, Dls_sum[cross_spec_key], color='black', label="mean")
        try:
            axs[row, col].errorbar(data_Planck[0], data_Planck[1]*fac(data_Planck[0]), data_Planck[2]*fac(data_Planck[0]), color='crimson', label="Planck", linestyle='None', marker='.')
        except:
            pass
        axs[row, col].set_ylim(spec_ylims[cross_spec_key])
    axs[row, col].set_xlim(spec_xlims[cross_spec_key])
    axs[row, col].legend()

plt.savefig(filename, dpi=300)

