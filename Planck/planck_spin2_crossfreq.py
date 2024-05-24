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
import datetime

def fac(ls):
    return ls * (ls + 1) / (2 * pi)

date_float = datetime.datetime.now().strftime("%m%d_%H%M")
print("run %s" % date_float)

FSKY = "Likelihood"
BAND_1 = "100"
BAND_2 = "100"

#The code takes filename.format(BAND_1, BAND_2)
filename = "Planck/Figures/Planck_{}x{}_full_bmcm.png" 
filename = "Planck/Figures/Planck_{}x{}_full_bmcm.png" 

plot_bool = False
save_spectra_bool = True
load_Planck_spectra_bool = False
same_fig_bool = False
err_plot_bool = False
comp_plot_bool = False
binned_mcm = False
adapt_lmax = False
start_at_2_bool = False
sum_cross_spec_bool = False

LMAX= 2000

BAND_LIST = ["100", "143", "217"]
# BAND_LIST = ["100"]
# BAND_LIST = ["217"]

# BAND_iter = [("143", "100"), ("217", "100"), ("217", "143")]
# BAND_iter = [("100", "100"), ("100", "143"), ("100", "217"), ("143", "143"), ("143", "217"), ("217", "217")]
# BAND_iter = combinations_with_replacement(BAND_LIST, r=2)    #Sans repeter xy et yx
BAND_iter = product(BAND_LIST, repeat=2)      #Tout

bin_size = 20
# pspy_utils.create_binning_file(bin_size, 10000/bin_size, 10000, "binning.dat")
binning_file = "data/spectra/planck/binning/bin_planck.dat"
# binning_file = "binning.dat"

fskys = [0.20, 0.40, 0.60, 0.70, 0.80, 0.90, 0.97, 0.99]
spec_keys = ["TT", "EE", "BB", "TE", "EB", "TB"]
spec_indices = [0, 3, 5, 3, 5, 1, 4, 4, 2]
spec_keys_pspy = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
color_band = {"100" : "red", 
                "143" : "green",
                "217" : "purple"}

spec_keys_to_plot = ["TT", "EE", "BB", "TE", 'TB', 'EB', 'ET', 'BT', 'BE']
# spec_keys_to_plot = ['TT', 'TE', 'EE']
# spec_keys_to_plot = ['EB']

rows, cols = best_subplot_shape(len(spec_keys_to_plot))

if same_fig_bool:
    ### Creating the figure
    fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*6), dpi=300)
    axs = np.array(axs).reshape(rows, cols)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust the spacing between subplots
    plt.tight_layout()

for BAND_1, BAND_2 in BAND_iter:
    filename = filename.format(BAND_1, BAND_2)
    lmax_iter = LMAX
    
    if not same_fig_bool:
        ### Creating the figure
        fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*6))
        axs = np.array(axs).reshape(rows, cols)
        fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust the spacing between subplots
        plt.tight_layout()
    
    print(BAND_1, BAND_2)
    print("Reading data")
    
    ### Trying to get Planck spectra and adapting LMAX if necessary
    if load_Planck_spectra_bool:
        planck_key_bool = {}
        data_Planck = {}
        for key in spec_keys_to_plot:
            try:
                data_Planck[key] = np.loadtxt("data/spectra/planck/planck_spectrum_%s_%sx%s.dat" % (key, BAND_1, BAND_2)).T
                if adapt_lmax:
                    lmax_iter = max(int(max(data_Planck[key][0]))+100, lmax_iter)
            except:
                pass

    ### Reading CMB maps
    map_1 = so_map.read_map(
        "data/maps/HFI_SkyMap_%s_2048_R3.01_halfmission-1.fits" % BAND_1, fields_healpix=(0, 1, 2)
    )
    map_1.data *= 1e6

    map_2 = so_map.read_map(
        "data/maps/HFI_SkyMap_%s_2048_R3.01_halfmission-2.fits" % BAND_2, fields_healpix=(0, 1, 2)
    )
    map_2.data *= 1e6
    
    print("Masks")
    ### Reading masks maps
    mask_pol_1 = so_map.read_map(
        "data/maps/COM_Mask_Likelihood-polarization-%s-hm1_2048_R3.00.fits" % BAND_1
    )
    mask_pol_2 = so_map.read_map(
        "data/maps/COM_Mask_Likelihood-polarization-%s-hm2_2048_R3.00.fits" % BAND_2
    )
    mask_T_1 = so_map.read_map(
        "data/maps/COM_Mask_Likelihood-temperature-%s-hm1_2048_R3.00.fits" % BAND_1
    )
    mask_T_2 = so_map.read_map(
        "data/maps/COM_Mask_Likelihood-temperature-%s-hm2_2048_R3.00.fits" % BAND_2
    )
    
    ### Substract monopole and dipole
    map_1.subtract_mono_dipole(mask=(mask_T_1, mask_pol_1))
    map_2.subtract_mono_dipole(mask=(mask_T_2, mask_pol_2))
    
    ### Compute spectra from masked maps
    almsList_1 = sph_tools.get_alms(map_1, (mask_T_1, mask_pol_1), niter=0, lmax=lmax_iter)
    almsList_2 = sph_tools.get_alms(map_2, (mask_T_2, mask_pol_2), niter=0, lmax=lmax_iter)
    ls_12, cls_12 = so_spectra.get_spectra(almsList_1, almsList_2, spectra=spec_keys_pspy)
    
    if sum_cross_spec_bool:
        cls_12['TE'] += cls_12['ET']
        cls_12['TE'] /= 2
        cls_12['ET'] += cls_12['TE']
        cls_12['ET'] /= 2
        cls_12['EB'] += cls_12['BE']
        cls_12['EB'] /= 2
        cls_12['BE'] += cls_12['EB']
        cls_12['BE'] /= 2
        cls_12['TB'] += cls_12['BT']
        cls_12['TB'] /= 2
        cls_12['BT'] += cls_12['TB']
        cls_12['BT'] /= 2

    ### Beam correction
    if BAND_1 == BAND_2:
        data_beam_T = np.loadtxt("data/beam_legacy/bl_T_legacy_%shm1x%shm2.dat" % (BAND_1, BAND_2)).T
        ls_beam_T, bls_T = data_beam_T[0, :len(ls_12)], data_beam_T[1, :len(ls_12)]

        data_beam_pol = np.loadtxt("data/beam_legacy/bl_pol_legacy_%shm1x%shm2.dat" % (BAND_1, BAND_2)).T
        ls_beam_pol, bls_pol = data_beam_pol[0, :len(ls_12)], data_beam_pol[1, :len(ls_12)]

        Bls_dict = {'TT' : bls_T**2,
                    'EE' : bls_pol**2,
                    'BB' : bls_pol**2,
                    'TE' : bls_pol*bls_T,
                    'ET' : bls_pol*bls_T,
                    'EB' : bls_pol**2,
                    'BE' : bls_pol**2,
                    'TB' : bls_pol*bls_T,
                    'BT' : bls_pol*bls_T}
    else:
        data_beam_T_1 = np.loadtxt("data/beam_legacy/bl_T_legacy_%shm1x%shm2.dat" % (BAND_1, BAND_1)).T
        bls_T_1 = data_beam_T_1[1, :len(ls_12)]

        data_beam_pol_1 = np.loadtxt("data/beam_legacy/bl_pol_legacy_%shm1x%shm2.dat" % (BAND_1, BAND_1)).T
        bls_pol_1 = data_beam_pol_1[1, :len(ls_12)]
        
        data_beam_T_2 = np.loadtxt("data/beam_legacy/bl_T_legacy_%shm1x%shm2.dat" % (BAND_2, BAND_2)).T
        bls_T_2 = data_beam_T_2[1, :len(ls_12)]

        data_beam_pol_2 = np.loadtxt("data/beam_legacy/bl_pol_legacy_%shm1x%shm2.dat" % (BAND_2, BAND_2)).T
        bls_pol_2 = data_beam_pol_2[1, :len(ls_12)]

        Bls_dict = {'TT' : bls_T_1 * bls_T_2,
                    'EE' : bls_pol_1 * bls_pol_2,
                    'BB' : bls_pol_1 * bls_pol_2,
                    'TE' : bls_T_1 * bls_pol_2,
                    'ET' : bls_pol_1 * bls_T_2,
                    'EB' : bls_pol_1 * bls_pol_2,
                    'BE' : bls_pol_1 * bls_pol_2,
                    'TB' : bls_T_1 * bls_pol_2,
                    'BT' : bls_pol_1 * bls_T_2}

    ### Mode coupling matrix
    mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2((mask_T_1, mask_pol_1), binning_file, lmax=lmax_iter, niter=0, type='Dl',
                                                win2=(mask_T_2, mask_pol_2), binned_mcm=binned_mcm)

    ### Pixwin correction
    pixwin_1 = map_1.get_pixwin()[:len(ls_12)]
    pixwin_2 = map_2.get_pixwin()[:len(ls_12)]

    ### Getting dict with the right keys for pspy
    cls_dict_beam = {
        spec_keys[i]: cls_12[spec_keys[i]]
        for i in range(len(spec_keys))
    }    
    
    for i, key in enumerate(spec_keys_pspy):
        key_ini = spec_keys[spec_indices[i]]
        cls_dict_beam[key] = cls_dict_beam[key_ini] / pixwin_1 / pixwin_2 / Bls_dict[key]

    ### Binning and mcm
    lb, cls_dict_bin = so_spectra.bin_spectra(
        ls_12,
        cls_dict_beam,
        binning_file,
        lmax=lmax_iter,
        type="Dl",
        mbb_inv=mbb_inv,
        spectra=spec_keys_pspy,
        binned_mcm=binned_mcm
    )
    
    ### Computing error
    if err_plot_bool:
        error_dict = get_cosmic_variance(cls_dict_bin, ls=lb)

    ### Saving spectra for later
    if save_spectra_bool:
        so_spectra.write_ps("data/spectra/maison/Dls_%sx%s_%s.dat" % (BAND_1, BAND_2, date_float), lb, cls_dict_bin, 'Dl', spectra=cls_dict_bin.keys())
    
    if plot_bool:
        print("Plotting")
        ### Plotting over wanted keys
        row, col = 0, 0
        for i, key in enumerate(spec_keys_to_plot):
            row = i // cols
            col = i % cols
            axs[row, col].set_title(key)

            if not comp_plot_bool:
                axs[row, col].plot(lb, cls_dict_bin[key], label="%sx%s" % (BAND_1, BAND_2), color=color_band[BAND_1])
                if key in ["TT", "EE", "BB"]:
                    axs[row, col].semilogy()

                if key in data_Planck.keys():
                    axs[row, col].errorbar(
                        data_Planck[key][0],
                        data_Planck[key][1] * fac(data_Planck[key][0]),
                        data_Planck[key][2] * fac(data_Planck[key][0]),
                        linestyle='None',
                        marker=".",
                        color="black",
                        alpha=0.8,
                        label="Planck release",
                    )
            if comp_plot_bool:
                try:
                    comp_indices = [i for i, value in enumerate(lb) if value in set(data_Planck[key][0])]
                    cls_comp = [cls_dict_bin[key][i] for i in comp_indices]
                    axs[row, col].plot(data_Planck[key][0],
                                        (cls_comp - (data_Planck[key][1] * fac(data_Planck[key][0]))) / (data_Planck[key][2] * fac(data_Planck[key][0])),
                                        label="%sx%s" % (BAND_1, BAND_2))  #, color=color_band[BAND_1]
                except:
                    pass

            if err_plot_bool:
                axs[row, col].fill_between(
                    lb,
                    cls_dict_bin[key] - error_dict[key] / np.sqrt(bin_size),
                    cls_dict_bin[key] + error_dict[key] / np.sqrt(bin_size),
                    alpha=0.3,
                    color="grey",
                    label="Effective noise",
                    zorder=-10,
                )
            axs[row, col].legend()
        if not same_fig_bool:
            # plt.grid()
            plt.savefig(filename)
            plt.clf()

if same_fig_bool:
    # plt.grid()
    # plt.ylim(-2, 2)
    plt.savefig(filename)
    plt.clf()

