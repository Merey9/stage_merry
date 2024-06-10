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
        map_1 = so_map.read_map(
            "data/maps/HFI_SkyMap_%s_2048_R3.01_halfmission-%s.fits" % (BAND_1, hm_B1),
            fields_healpix=(0, 1, 2),
        )
        map_1.data *= 1e6

        map_2 = so_map.read_map(
            "data/maps/HFI_SkyMap_%s_2048_R3.01_halfmission-%s.fits" % (BAND_2, hm_B2),
            fields_healpix=(0, 1, 2),
        )
        map_2.data *= 1e6

        # print("Masks")
        ### Reading masks maps
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

        #    lb_1, cls_dict_bin_1 = so_spectra.bin_spectra(
        #        ls_11,
        #        cls_dict_pspy_1,
        #        binning_file,
        #        lmax=lmax_iter,
        #        type="Dl",
        #        mbb_inv=mbb_inv,
        #        spectra=spec_keys_pspy,
        #        binned_mcm=binned_mcm,
        #    )
        #
        #    lb_2, cls_dict_bin_2 = so_spectra.bin_spectra(
        #        ls_22,
        #        cls_dict_pspy_2,
        #        binning_file,
        #        lmax=lmax_iter,
        #        type="Dl",
        #        mbb_inv=mbb_inv,
        #        spectra=spec_keys_pspy,
        #        binned_mcm=binned_mcm,
        #    )

        # assert len(ls_12) == len(ls_11)

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
        #     so_spectra.write_ps(
        #         "data/spectra/maison/Dls_%sx%s_%s_11.dat" % (BAND_1, BAND_2, date_float),
        #         lb,
        #         cls_dict_bin_1,
        #         "Dl",
        #         spectra=cls_dict_bin_1.keys(),
        #     )
        #     so_spectra.write_ps(
        #         "data/spectra/maison/Dls_%sx%s_%s_22.dat" % (BAND_1, BAND_2, date_float),
        #         lb,
        #         cls_dict_bin_2,
        #         "Dl",
        #         spectra=cls_dict_bin_2.keys(),
        #     )

    # print(cls_dict_bin.keys())

    if plot_bool:
        print("Plotting")
        ### Plotting over wanted keys
        row, col = 0, 0
        for i, key in enumerate(spec_keys_to_plot):
            row = i // cols
            col = i % cols
            axs[row, col].set_title(key)

            if not comp_plot_bool:
                axs[row, col].plot(
                    lb,
                    cls_dict_bin[key],
                    label="%sx%s" % (BAND_1, BAND_2),
                    color=color_band[BAND_1],
                )
                if key in ["TT", "EE", "BB"]:
                    axs[row, col].semilogy()

                if key in data_Planck.keys():
                    axs[row, col].errorbar(
                        data_Planck[key][0],
                        data_Planck[key][1] * fac(data_Planck[key][0]),
                        data_Planck[key][2] * fac(data_Planck[key][0]),
                        linestyle="None",
                        marker=".",
                        color="black",
                        alpha=0.8,
                        label="Planck release",
                    )
            if comp_plot_bool:
                try:
                    comp_indices = [
                        i
                        for i, value in enumerate(lb)
                        if value in set(data_Planck[key][0])
                    ]
                    cls_comp = [cls_dict_bin[key][i] for i in comp_indices]
                    axs[row, col].plot(
                        data_Planck[key][0],
                        (cls_comp - (data_Planck[key][1] * fac(data_Planck[key][0])))
                        / (data_Planck[key][2] * fac(data_Planck[key][0])),
                        label="%sx%s" % (BAND_1, BAND_2),
                    )  # , color=color_band[BAND_1]
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
