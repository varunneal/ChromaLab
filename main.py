from typing import List, Tuple, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import numpy as np
import numpy.typing as npt

from itertools import combinations


from spectra import Spectra, Pigment


def mix(pigment1: Pigment, pigment2: Pigment, concentrations: Optional[List[float]] = None) -> Pigment:
    assert np.array_equal(pigment1.reflectance[:, 0], pigment2.reflectance[:, 0])
    wavelengths = pigment1.wavelengths()

    # todo: interpolate/sample so the wavelength data can be different
    # potential implementation: pass in wavelengths to get_k_s,
    # and then the result will be two lists of length agreeing with len(wavelengths)

    if not concentrations:
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
    def __init__(self, sensors: List[Spectra]):
        self.dimension = len(sensors)
        self.sensors = sensors

        total_wavelengths = np.unique(np.concatenate([sensor.wavelengths() for sensor in sensors]))
        num_wavelengths = len(total_wavelengths)

        self.sensor_matrix = np.zeros((self.dimension, num_wavelengths))

        for i, sensor in enumerate(sensors):
            for j, wavelength in enumerate(total_wavelengths):
                self.sensor_matrix[i, j] = sensor.interpolated_value(wavelength)

    def get_full_colors(self) -> npt.NDArray:
        # calculates the hull of the gamut using schrodinger's transition functions for optimal colors
        num_wavelengths = self.sensor_matrix.shape[1]

        transitions = []
        min_transition_size = 10
        indices = list(range(num_wavelengths // min_transition_size))

        for selected_indices in combinations(indices, 2 * (self.dimension - 1)):
            sorted_indices = sorted(selected_indices)
            intervals = [(sorted_indices[i], sorted_indices[i + 1]) for i in range(0, len(sorted_indices), 2)]

            transition = np.zeros(num_wavelengths)
            for start, end in intervals:
                transition[min_transition_size * start:min_transition_size * end] = 1

            transitions.append(transition)

        print(f"{len(transitions)} full colors generated.")

        transitions_matrix = np.array(transitions).T

        return np.matmul(self.sensor_matrix, transitions_matrix)


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

    # s_cone.plot(name="s", color="b")
    # m_cone.plot(name="m", color="g")
    # l_cone.plot(name="l", color="r")
    # plt.xlabel('wavelength (nm)')
    # plt.ylabel('spectral reflectance')
    # plt.title('Cones')
    # plt.legend()
    # plt.show()

    trichromat = Observer([s_cone, m_cone, l_cone])
    print(trichromat.sensor_matrix.shape)
    full_colors_3d = trichromat.get_full_colors()
    print(full_colors_3d.shape)

    x,y,z = full_colors_3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, c='blue', marker='o', alpha=0.5)  # s is the size of points, adjust as needed

    # Labels and title
    ax.set_xlabel('s')
    ax.set_ylabel('m')
    ax.set_zlabel('l')
    ax.set_title('Trichromat Full Color Gamut')
    ax.view_init(elev=30, azim=30)
    plt.show()


    """
    Kubelka-Munk color mixing
    """
    wavelengths = np.arange(400, 701)

    yellow_curve = np.exp(-0.5 * ((wavelengths - 450) / 30) ** 2)
    magenta_curve = np.exp(-0.5 * ((wavelengths - 550) / 30) ** 2)

    yellow = Pigment(Spectra(wavelengths=wavelengths, data=yellow_curve))
    magenta = Pigment(Spectra(wavelengths=wavelengths, data=magenta_curve))

    concentration_range = np.linspace(0, 1, 11)  # For example, 11 steps from 0.0 to 1.0
    yellow_rgb = np.array(to_rgba('yellow')[:3])
    magenta_rgb = np.array(to_rgba('magenta')[:3])

    for c in concentration_range:
        mix_concentration = [c, 1 - c]
        mixed_pigment = mix(yellow, magenta, concentrations=mix_concentration)

        interp_color = c * yellow_rgb + (1 - c) * magenta_rgb

        alpha_value = 0.75
        if not c % 0.2:
            alpha_value = 1.0

        # Convert the interpolated RGB to a matplotlib-readable format
        interp_color_str = f'#{int(interp_color[0] * 255):02X}{int(interp_color[1] * 255):02X}{int(interp_color[2] * 255):02X}'

        # mixed_pigment.plot(name="", color=interp_color_str, alpha=alpha_value)

    # yellow.plot(name='yellow', color='yellow')
    # magenta.plot(name='magenta', color='magenta')
    # plt.legend()
    # plt.show()


main()
