import numpy as np
from math import cos, sin

list_keys = ["TT", "EE", "BB", "TE", "TB", "EB"]


def biref_cross_spectra(
    cls, angle_params={"alpha": 0, "beta": 0}, wanted_keys=["TT", "EE", "BB", "TE"]
):
    alpha = angle_params["alpha"]
    beta = angle_params["beta"]

    cls_full = cls.copy()
    for key in cls_full:
        cls_full[key] = np.array(cls_full[key])
    for missing_key in [key for key in list_keys if key not in cls.keys()]:
        cls_full[missing_key] = np.zeros_like(cls_full["TT"], dtype=float)

    cls_calc = {}
    cls_calc["TT"] = cls_full["TT"]
    cls_calc["EE"] = (
        cls_full["EE"] * cos(2 * alpha) ** 2
        + cls_full["BB"] * sin(2 * alpha) ** 2
        - sin(4 * alpha) * cls_full["EB"]
    )
    cls_calc["BB"] = (
        cls_full["EE"] * sin(2 * alpha) ** 2
        + cls_full["BB"] * cos(2 * alpha) ** 2
        + sin(4 * alpha) * cls_full["EB"]
    )
    cls_calc["TE"] = cls_full["TE"] * cos(2 * alpha) - cls_full["TB"] * sin(2 * alpha)
    cls_calc["TB"] = cls_full["TE"] * sin(2 * alpha) + cls_full["TB"] * cos(2 * alpha)
    cls_calc["EB"] = (cls_full["EE"] - cls_full["BB"]) * sin(4 * alpha) / 2 + cls_full[
        "EB"
    ] * cos(4 * alpha)

    cls_return = {}
    for key in wanted_keys:
        cls_return[key] = cls_calc[key]
    return cls_return


def cls_1dev_alpha(cls, alpha):
    cls_full = cls.copy()
    for key in cls_full:
        cls_full[key] = np.array(cls_full[key])
    for missing_key in [key for key in list_keys if key not in cls.keys()]:
        cls_full[missing_key] = np.zeros_like(cls_full["TT"], dtype=float)

    cls_1dev = {}
    cls_1dev["TT"] = cls_full["TT"]
    cls_1dev["EE"] = (cls_full["BB"] - cls_full["EE"]) * 2 * sin(4 * alpha)
    cls_1dev["BB"] = (cls_full["EE"] - cls_full["BB"]) * 2 * sin(4 * alpha)
    cls_1dev["TE"] = -2 * cls_full["TE"] * sin(2 * alpha) + 2 * cls_full["TB"] * cos(
        2 * alpha
    )
    cls_1dev["TB"] = -2 * cls_full["TB"] * sin(2 * alpha) + 2 * cls_full["TE"] * cos(
        2 * alpha
    )
    cls_1dev["EB"] = (cls_full["EE"] - cls_full["BB"]) * 2 * cos(4 * alpha) - cls_full[
        "EB"
    ] * 4 * sin(4 * alpha)

    return cls_1dev


# print(biref_cross_spectra({'TT':np.array([0,0,0,0]), 'EE':[1,2,3,4]}, angle_params={'alpha':-1e-2, 'beta':0}))
