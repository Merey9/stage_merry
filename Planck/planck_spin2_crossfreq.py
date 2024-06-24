import sys

sys.path.append("/data/stage_merry/functions")
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
# for beta_deg in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1., 1.5]:
# date_float += '_beta%s' % (beta_deg)
print("run %s" % date_float)
# date_float = "0530_0810"

FSKY = "Likelihood"
BAND_1 = "100"
BAND_2 = "100"

# The code takes filename.format(BAND_1, BAND_2)
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
get_fskys = False

which_maps = "Planck_DR4"  # Planck_legacy,  biref_0.1..., Planck_DR4
which_mask = "biref"  # Planck,  biref ...

niter = 3

LMAX = 3000

BAND_LIST = ["100", "143", "217"]
# BAND_LIST = ["100"]
# BAND_LIST = ["217"]

# BAND_iter = [("143", "100"), ("217", "100"), ("217", "143")]
# BAND_iter = [("100", "100"), ("100", "143"), ("100", "217"), ("143", "143"), ("143", "217"), ("217", "217")]
# BAND_iter = combinations_with_replacement(BAND_LIST, r=2)    #Sans repeter xy et yx
BAND_iter = product(BAND_LIST, repeat=2)  # Tout
# BAND_iter = [("143", "100"), ("100", "143")]

bin_size = 20
new_binning_file = "data/spectra/maison/binning/binning%s.dat" % date_float
pspy_utils.create_binning_file(bin_size, 10000 / bin_size, 10000, new_binning_file)
# binning_file = "data/spectra/planck/binning/bin_planck.dat"
binning_file = new_binning_file

fskys = [0.20, 0.40, 0.60, 0.70, 0.80, 0.90, 0.97, 0.99]
spec_keys = ["TT", "EE", "BB", "TE", "EB", "TB"]
spec_indices = [0, 3, 5, 3, 5, 1, 4, 4, 2]
spec_keys_pspy = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
color_band = {"100": "red", "143": "green", "217": "purple"}

spec_keys_to_plot = ["TT", "EE", "BB", "TE", "TB", "EB", "ET", "BT", "BE"]
# spec_keys_to_plot = ['TT', 'TE', 'EE']
# spec_keys_to_plot = ['EB']

rows, cols = best_subplot_shape(len(spec_keys_to_plot))

if same_fig_bool:
    ### Creating the figure
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), dpi=300)
    axs = np.array(axs).reshape(rows, cols)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust the spacing between subplots
    plt.tight_layout()

if which_mask == "biref":
    mask_pol_1 = so_map.read_map("data/maps/comb_0_mask_hfi_lfi_nside_2048.fits")
    mask_pol_2 = mask_pol_1.copy()
    mask_T_1 = mask_pol_1.copy()
    mask_T_2 = mask_pol_1.copy()

for BAND_1, BAND_2 in tqdm(BAND_iter):
    filename = filename.format(BAND_1, BAND_2)
    lmax_iter = LMAX

    if not same_fig_bool:
        ### Creating the figure
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
        axs = np.array(axs).reshape(rows, cols)
        fig.subplots_adjust(
            hspace=0.2, wspace=0.2
        )  # Adjust the spacing between subplots
        plt.tight_layout()

    # print(BAND_1, BAND_2)
    # print("Reading data")
    for hm_B1, hm_B2 in [("1", "1"), ("1", "2"), ("2", "1"), ("2", "2")]:
        # for hm_B1, hm_B2 in [('2', '1')]:
        # print(hm_B1 + hm_B2)
        # if BAND_1 == BAND_2 and (hm_B1, hm_B2) == ('2', '1'):
        #  continue
        ### Trying to get Planck spectra and adapting LMAX if necessary
        if load_Planck_spectra_bool:
            planck_key_bool = {}
            data_Planck = {}
            for key in spec_keys_to_plot:
                try:
                    data_Planck[key] = np.loadtxt(
                        "data/spectra/planck/planck_spectrum_%s_%sx%s.dat"
                        % (key, BAND_1, BAND_2)
                    ).T
                    if adapt_lmax:
                        lmax_iter = max(int(max(data_Planck[key][0])) + 100, lmax_iter)
                except:
                    pass

        ### Reading CMB maps
        if which_maps == "Planck_legacy":
            map_1 = so_map.read_map(
                "data/maps/HFI_SkyMap_%s_2048_R3.01_halfmission-%s.fits"
                % (BAND_1, hm_B1),
                fields_healpix=(0, 1, 2),
            )

            map_2 = so_map.read_map(
                "data/maps/HFI_SkyMap_%s_2048_R3.01_halfmission-%s.fits"
                % (BAND_2, hm_B2),
                fields_healpix=(0, 1, 2),
            )

        if which_maps == "Planck_DR4":
            map_1 = so_map.read_map(
                "data/maps/HFI_SkyMap_%s-BPassCorrected_2048_R4.00_full-ringhalf-%s.fits"
                % (BAND_1, hm_B1),
                fields_healpix=(0, 1, 2),
            )

            map_2 = so_map.read_map(
                "data/maps/HFI_SkyMap_%s-BPassCorrected_2048_R4.00_full-ringhalf-%s.fits"
                % (BAND_2, hm_B2),
                fields_healpix=(0, 1, 2),
            )

        if which_maps == "simu_biref_1":
            map_1 = so_map.read_map(
                "data/maps/maison/LCDM_biref_%s_%s_hm%s.fits"
                % (beta_deg, BAND_1, hm_B1),
                fields_healpix=(0, 1, 2),
            )

            map_2 = so_map.read_map(
                "data/maps/maison/LCDM_biref_%s_%s_hm%s.fits"
                % (beta_deg, BAND_2, hm_B2),
                fields_healpix=(0, 1, 2),
            )

        # print("Masks")
        ### Reading masks maps
        if which_mask == "Planck":
            mask_pol_1 = so_map.read_map(
                "data/maps/COM_Mask_Likelihood-polarization-%s-hm%s_2048_R3.00.fits"
                % (BAND_1, hm_B1)
            )
            mask_pol_2 = so_map.read_map(
                "data/maps/COM_Mask_Likelihood-polarization-%s-hm%s_2048_R3.00.fits"
                % (BAND_2, hm_B2)
            )
            mask_T_1 = so_map.read_map(
                "data/maps/COM_Mask_Likelihood-temperature-%s-hm%s_2048_R3.00.fits"
                % (BAND_1, hm_B1)
            )
            mask_T_2 = so_map.read_map(
                "data/maps/COM_Mask_Likelihood-temperature-%s-hm%s_2048_R3.00.fits"
                % (BAND_2, hm_B2)
            )

        if get_fskys:
            print(so_window.get_survey_solid_angle(mask_T_1) / (4 * pi))
            print(so_window.get_survey_solid_angle(mask_T_2) / (4 * pi))
            print(so_window.get_survey_solid_angle(mask_pol_1) / (4 * pi))
            print(so_window.get_survey_solid_angle(mask_pol_1) / (4 * pi))
            continue

        ### Substract monopole and dipole
        map_1.subtract_mono_dipole(mask=(mask_T_1, mask_pol_1))
        map_2.subtract_mono_dipole(mask=(mask_T_2, mask_pol_2))

        map_1.data *= 1e6
        map_2.data *= 1e6
        ### Compute spectra from masked maps
        almsList_1 = sph_tools.get_alms(
            map_1, (mask_T_1, mask_pol_1), niter=niter, lmax=lmax_iter
        )
        almsList_2 = sph_tools.get_alms(
            map_2, (mask_T_2, mask_pol_2), niter=niter, lmax=lmax_iter
        )
        ls_12, cls_12 = so_spectra.get_spectra(
            almsList_1, almsList_2, spectra=spec_keys_pspy
        )
        #        ls_11, cls_11 = so_spectra.get_spectra(
        #            almsList_1, almsList_1, spectra=spec_keys_pspy
        #        )
        #        ls_22, cls_22 = so_spectra.get_spectra(
        #            almsList_2, almsList_2, spectra=spec_keys_pspy
        #        )
        #        ls_21, cls_21 = so_spectra.get_spectra(
        #            almsList_1, almsList_2, spectra=spec_keys_pspy
        #         )

        data_beam_T_1 = np.loadtxt(
            "data/beam_legacy/bl_T_legacy_%shm%sx%shm%s.dat"
            % (BAND_1, hm_B1, BAND_1, hm_B1)
        ).T
        bls_T_1 = data_beam_T_1[1, : lmax_iter + 2]

        data_beam_pol_1 = np.loadtxt(
            "data/beam_legacy/bl_pol_legacy_%shm%sx%shm%s.dat"
            % (BAND_1, hm_B1, BAND_1, hm_B1)
        ).T
        bls_pol_1 = data_beam_pol_1[1, : lmax_iter + 2]

        data_beam_T_2 = np.loadtxt(
            "data/beam_legacy/bl_T_legacy_%shm%sx%shm%s.dat"
            % (BAND_2, hm_B2, BAND_2, hm_B2)
        ).T
        bls_T_2 = data_beam_T_2[1, : lmax_iter + 2]

        data_beam_pol_2 = np.loadtxt(
            "data/beam_legacy/bl_pol_legacy_%shm%sx%shm%s.dat"
            % (BAND_2, hm_B2, BAND_2, hm_B2)
        ).T
        bls_pol_2 = data_beam_pol_2[1, : lmax_iter + 2]

        ### Mode coupling matrix
        mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(
            (mask_T_1, mask_pol_1),
            binning_file,
            lmax=lmax_iter,
            niter=niter,
            type="Dl",
            win2=(mask_T_2, mask_pol_2),
            binned_mcm=binned_mcm,
            bl1=(bls_T_1, bls_pol_1),
            bl2=(bls_T_2, bls_pol_2),
        )

        ### Pixwin correction
        pixwin_1 = map_1.get_pixwin()[: len(ls_12)]
        pixwin_2 = map_2.get_pixwin()[: len(ls_12)]

        ### Getting dict with the right keys for pspy
        cls_dict_pspy = {
            key: cls_12[key] / pixwin_1 / pixwin_2 for key in cls_12.keys()
        }

        # cls_dict_pspy_1 = {key: cls_11[key] / pixwin_1 / pixwin_1 for key in cls_11.keys()}

        # cls_dict_pspy_2 = {key: cls_22[key] / pixwin_2 / pixwin_2 for key in cls_22.keys()}
        #
        ### Binning and mcm
        lb, cls_dict_bin = so_spectra.bin_spectra(
            ls_12,
            cls_dict_pspy,
            binning_file,
            lmax=lmax_iter,
            type="Dl",
            mbb_inv=mbb_inv,
            spectra=spec_keys_pspy,
            binned_mcm=binned_mcm,
        )

        ### Computing error
        if err_plot_bool:
            error_dict = get_cosmic_variance(cls_dict_bin, ls=lb)

        ### Saving spectra for later
        if save_spectra_bool:
            so_spectra.write_ps(
                "data/spectra/maison/Dls_%shm%sx%shm%s_%s.dat"
                % (BAND_1, hm_B1, BAND_2, hm_B2, date_float),
                lb,
                cls_dict_bin,
                "Dl",
                spectra=cls_dict_bin.keys(),
            )
