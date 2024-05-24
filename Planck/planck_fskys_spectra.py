import healpy as hp
from pspy import *
from matplotlib import pyplot as plt
import numpy as np
from math import pi

LMAX = 2500

BAND = "143"

fskys = [0.20, 0.40, 0.60, 0.70, 0.80, 0.90, 0.97, 0.99]


spec_keys = ["TT", "EE", "BB", "TE", "EB", "TB"]

map_1 = hp.fitsfunc.read_map(
    "data/maps/HFI_SkyMap_%s_2048_R3.01_halfmission-1.fits" % BAND, field=(0, 1, 2)
)

map_2 = hp.fitsfunc.read_map(
    "data/maps/HFI_SkyMap_%s_2048_R3.01_halfmission-2.fits" % BAND, field=(0, 1, 2)
)
cls_array = hp.anafast(map_1, map_2, lmax=LMAX)
cls_dict = {spec_keys[i]: cls_array[i][:LMAX] for i in range(len(spec_keys))}

ls = np.arange(2, LMAX)
fac = ls * (ls + 1) / (2 * pi)

pspy_utils.create_binning_file(20, 125, 2500, "binning.dat")

cls_dict_bin = {}
cls_dict_bin_deconv = {}
for i, key in enumerate(["TT", "EE", "BB", "TE", "EB", "TB"]):
    for mask_field in range(len(fskys)):
        FSKY = fskys[mask_field]
        so_map_apo0 = so_map.read_map(
            "data/maps/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",
            fields_healpix=mask_field,
        )

        map_1_mask = map_1 * so_map_apo0.data
        map_2_mask = map_2 * so_map_apo0.data

        cls_array_mask = hp.anafast(map_1_mask, map_2_mask, lmax=LMAX)
        cls_dict_mask = {
            spec_keys[i]: cls_array_mask[i][:LMAX] for i in range(len(spec_keys))
        }

        mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0(
            so_map_apo0, "binning.dat", lmax=2000, niter=0
        )

        lb, cls_dict_bin_deconv[key] = so_spectra.bin_spectra(
            ls, cls_dict_mask[key], "binning.dat", lmax=2000, type="Dl", mbb_inv=mbb_inv
        )
        plt.plot(lb, cls_dict_bin_deconv[key], label=BAND + " FSKY=" + str(FSKY))

    lb, cls_dict_bin[key] = so_spectra.bin_spectra(
        ls, cls_dict[key], "binning.dat", lmax=2000, type="Dl"
    )
    # plt.plot(ls, cls_dict[key][2:]*fac, label=BAND)
    plt.plot(lb, cls_dict_bin[key], label=BAND + " fullsky")
    plt.legend()
    if key in ["TT", "EE", "BB"]:
        plt.semilogy()
    plt.savefig("Figures/Planck_" + key + ".png")
    plt.clf()
