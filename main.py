from typing import List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from spectra import Spectra, Pigment


def mix(pigment1: Pigment, pigment2: Pigment) -> Pigment:
    assert np.array_equal(pigment1.reflectance[:, 0], pigment2.reflectance[:, 0])
    wavelengths = pigment1.wavelengths()

    # todo: interpolate/sample so the wavelength data can be different
    # potential implementation: pass in wavelengths to get_k_s,
    # and then the result will be two lists of length agreeing with len(wavelengths)

    concentrations = [0.5, 0.5]
    k_mix, s_mix = [], []
    for k1, s1, k2, s2 in zip(*pigment1.get_k_s(), *pigment2.get_k_s()):
        k_mix.append(concentrations[0] * k1 + concentrations[1] * k2)
        s_mix.append(concentrations[0] * s1 + concentrations[1] * s2)

    k = np.column_stack((wavelengths, k_mix))
    s = np.column_stack((wavelengths, s_mix))

    return Pigment(k=k, s=s)


class Cone(Spectra):
    def __init__(self, reflectance: npt.NDArray):
        super().__init__(reflectance)


class Observer:
    # todo: find complementary saturated points in Observer's gamut
    def __init__(self, sensors: List[Spectra]):
        self.dimension = len(sensors)
        self.sensors = sensors


def remove_trailing_nans(arr):
    mask = np.any(np.isnan(arr), axis=1)
    idx = np.where(mask)[0]
    if idx.size > 0:
        last_valid_idx = np.where(~mask)[0][-1]
        return arr[:last_valid_idx + 1]
    return arr


def main():
    """
    Human cone data
    """
    cone_data = np.genfromtxt('linss2_10e_1.csv', delimiter=',')

    s_cone = Cone(cone_data[:, [0, 1]])
    m_cone = Cone(cone_data[:, [0, 2]])
    l_cone = Cone(remove_trailing_nans(cone_data[:, [0, 3]]))

    s_cone.plot(name="s", color="b")
    m_cone.plot(name="m", color="g")
    l_cone.plot(name="l", color="r")
    plt.xlabel('wavelength (nm)')
    plt.ylabel('spectral reflectance')
    plt.title('Cones')
    plt.legend()
    plt.show()

    trichromat = Observer([s_cone, m_cone, l_cone])

    """
    Kubelka-Munk color mixing
    """
    wavelengths = np.arange(400, 701)

    yellow_curve = np.exp(-0.5 * ((wavelengths - 450) / 30) ** 2)
    magenta_curve = np.exp(-0.5 * ((wavelengths - 550) / 30) ** 2)

    yellow = Pigment(Spectra(wavelengths=wavelengths, data=yellow_curve))
    magenta = Pigment(Spectra(wavelengths=wavelengths, data=magenta_curve))

    mystery = mix(yellow, magenta)

    mystery.plot(name="mystery", color="black")
    yellow.plot(name='yellow', color='yellow')
    magenta.plot(name='magenta', color='magenta')
    plt.legend()
    plt.show()


main()
