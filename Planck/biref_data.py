from pspy import *
import camb
import healpy as hp
import numpy as np
from numpy import sin, cos
from copy import deepcopy
from matplotlib import pyplot as plt

maps = so_map.read_map("data/maps/maison/LCDM_01.fits")  #

LMAX = 2500
# Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
# This function sets up with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(LMAX, lens_potential_accuracy=0)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")

spectra = "total"

dls_TT = powers[spectra][:, 0]
dls_EE = powers[spectra][:, 1]
dls_BB = powers[spectra][:, 2]
dls_TE = powers[spectra][:, 3]

ls = np.arange(dls_TT.shape[0])
fac = ls * (ls + 1) / (2 * np.pi)

cls_TT = dls_TT / fac  # / (data_beam_T_1[:len(dls_TT)] ** 2)
cls_EE = dls_EE / fac  # / (data_beam_T_1[:len(dls_TT)] ** 2)
cls_TE = dls_TE / fac  # / (data_beam_T_1[:len(dls_TT)] ** 2)
cls_BB = dls_BB / fac  # / (data_beam_T_1[:len(dls_TT)] ** 2)

cls_TT[0:2] = 0
cls_EE[0:2] = 0
cls_TE[0:2] = 0
cls_BB[0:2] = 0

maps = hp.synfast((cls_TT, cls_TE, cls_EE, cls_BB), nside=2048)

for beta_deg in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5]:

    beta = beta_deg * 3.1415 / 180
    maps_biref = deepcopy(maps)
    map_T = maps[0]
    map_Q = maps[1]
    map_U = maps[2]
    maps_biref[1] = -map_U * sin(2 * beta) + map_Q * cos(2 * beta)
    maps_biref[2] = map_Q * sin(2 * beta) + map_U * cos(2 * beta)

    hp.fitsfunc.write_map(
        "data/maps/maison/LCDM_biref_%s.fits" % (beta_deg), maps_biref, overwrite=True
    )

    maps_so_biref = so_map.read_map("data/maps/maison/LCDM_biref_%s.fits" % (beta_deg))

    alms_biref = hp.map2alm(maps_so_biref.data, lmax=LMAX)

    for band in ["100", "143", "217"]:

        l_beam_T, data_beam_T = np.loadtxt(
            "data/beam_legacy/bl_T_legacy_%shm%sx%shm%s.dat" % (band, "1", band, "2")
        ).T
        l_beam_pol, data_beam_pol = np.loadtxt(
            "data/beam_legacy/bl_pol_legacy_%shm%sx%shm%s.dat" % (band, "1", band, "2")
        ).T

        beam_list = [
            data_beam_T[: LMAX + 1],
            data_beam_pol[: LMAX + 1],
            data_beam_pol[: LMAX + 1],
        ]

        alms_beam = [
            hp.sphtfunc.almxfl(alms_biref[i], beam_list[i])
            for i in range(len(alms_biref))
        ]

        maps_beam = hp.sphtfunc.alm2map(alms_beam, 2048)

        hp.fitsfunc.write_map(
            "data/maps/maison/LCDM_biref_%s_%s_beam.fits" % (beta_deg, band),
            maps_beam,
            overwrite=True,
        )
        maps_so_beam = so_map.read_map(
            "data/maps/maison/LCDM_biref_%s_%s_beam.fits" % (beta_deg, band)
        )

        noised_maps_hm1 = so_map.white_noise(maps_so_beam, 1.29 * 60, 1.96 * 60)
        noised_maps_hm2 = so_map.white_noise(maps_so_beam, 1.29 * 60, 1.96 * 60)

        maps_so_beam_hm1 = maps_so_beam.data + noised_maps_hm1.data
        maps_so_beam_hm2 = maps_so_beam.data + noised_maps_hm2.data

        hp.fitsfunc.write_map(
            "data/maps/maison/LCDM_biref_%s_%s_hm1.fits" % (beta_deg, band),
            maps_so_beam_hm1,
            overwrite=True,
        )
        hp.fitsfunc.write_map(
            "data/maps/maison/LCDM_biref_%s_%s_hm2.fits" % (beta_deg, band),
            maps_so_beam_hm2,
            overwrite=True,
        )
