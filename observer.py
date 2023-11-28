from itertools import combinations
from typing import List, Union, Optional

import numpy as np
import numpy.typing as npt

from spectra import Spectra, Pigment


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def remove_trailing_nans(arr):
    mask = np.any(np.isnan(arr), axis=1)
    idx = np.where(mask)[0]
    if idx.size > 0:
        last_valid_idx = np.where(~mask)[0][-1]
        return arr[:last_valid_idx + 1]
    return arr



class Cone(Spectra):
    cone_data = np.genfromtxt('linss2_10e_1.csv', delimiter=',')

    def __init__(self, reflectance: Optional[Union[Spectra, npt.NDArray]] = None,
                 wavelengths: Optional[npt.NDArray] = None,
                 data: Optional[npt.NDArray] = None):
        if isinstance(reflectance, Spectra):
            super().__init__(reflectance.reflectance)
        else:
            super().__init__(reflectance=reflectance, wavelengths=wavelengths, data=data)



    @staticmethod
    def l_cone(wavelengths=None):
        # 390 to 700
        reflectances = Spectra(Cone.cone_data[:311, [0,1]])
        return Cone(reflectances.interpolate_values(wavelengths))


    @staticmethod
    def q_cone(wavelengths=None):
        shift = 15
        m_cone = Cone.m_cone(wavelengths)
        r = [(w, 1e-4) for w in m_cone.wavelengths() if w < m_cone.wavelengths()[0] + shift] + \
            [(w + shift, v) for (w, v) in m_cone.reflectance if w + shift <= m_cone.wavelengths()[-1]]
        q_cone = Cone(reflectance=np.array(r))
        return q_cone

    @staticmethod
    def m_cone(wavelengths=None):
        # 390 to 700
        reflectances = Spectra(Cone.cone_data[:311, [0,2]])
        return Cone(reflectances.interpolate_values(wavelengths))


    @staticmethod
    def s_cone(wavelengths=None):
        # 390 to 700
        reflectances = Spectra(remove_trailing_nans(Cone.cone_data[:311, [0,3]]))
        return Cone(reflectances.interpolate_values(wavelengths))


class Observer:
    def __init__(self, sensors: List[Spectra], min_transition_size: int = 10):
        self.dimension = len(sensors)
        self.sensors = sensors

        total_wavelengths = np.unique(np.concatenate([sensor.wavelengths() for sensor in sensors]))
        self.wavelengths = total_wavelengths

        self.sensor_matrix = self.get_sensor_matrix(total_wavelengths)
        self.whitepoint = np.matmul(self.sensor_matrix, np.ones_like(self.wavelengths))

        self.min_transition_size = min_transition_size


    @staticmethod
    def trichromat(wavelengths=None):
        l_cone = Cone.l_cone(wavelengths)
        m_cone = Cone.m_cone(wavelengths)
        s_cone = Cone.s_cone(wavelengths)
        return Observer([s_cone, m_cone, l_cone])

    @staticmethod
    def tetrachromat(wavelengths=None):
        l_cone = Cone.l_cone(wavelengths)
        q_cone = Cone.q_cone(wavelengths)
        m_cone = Cone.m_cone(wavelengths)
        s_cone = Cone.s_cone(wavelengths)
        return Observer([s_cone, m_cone, q_cone, l_cone])


    def get_whitepoint(self, wavelengths: Optional[npt.NDArray] = None):
        # todo: differing illuminants
        sensor_matrix = self.get_sensor_matrix(wavelengths)

        return np.matmul(sensor_matrix, np.ones(sensor_matrix.shape[1]))


    def get_sensor_matrix(self, wavelengths: Optional[npt.NDArray] = None):
        """
        Get sensor matrix as specific wavelengths
        """
        if wavelengths is None:
            return self.sensor_matrix
        # assert wavelengths is 1d etc
        sensor_matrix = np.zeros((self.dimension, wavelengths.shape[0]))
        for i, sensor in enumerate(self.sensors):
            for j, wavelength in enumerate(wavelengths):
                sensor_matrix[i, j] = sensor.interpolated_value(wavelength)

        return sensor_matrix

    def observe(self, data: Union[npt.NDArray, Spectra]):
        """
        Either pass in a Spectra of light to observe or data that agrees with self.wavelengths.
        """
        if isinstance(data, Spectra):
            if np.array_equal(data.wavelengths(), self.wavelengths):
                data = data.data()
            else:
                # have to interpolate
                interp_color = []
                for wavelength in self.wavelengths:
                    interp_color.append(data.interpolated_value(wavelength))
                data = np.array(interp_color)
        else:
            assert data.size == self.wavelengths.size, f"Data shape {data.shape} must match wavelengths shape {self.wavelengths.shape}"
        observed_color = np.matmul(self.sensor_matrix, data)
        return np.divide(observed_color, self.whitepoint)

    def dist(self, color1: Union[npt.NDArray, Spectra], color2: Union[npt.NDArray, Spectra]):
        return np.linalg.norm(self.observe(color1) - self.observe(color2))

    def get_transitions(self) -> List[npt.NDArray]:
        transitions = []

        num_wavelengths = self.sensor_matrix.shape[1]
        indices = list(range(num_wavelengths // self.min_transition_size))

        # append black and white spectra at the start
        transitions.append(np.zeros(num_wavelengths))
        transitions.append(np.ones(num_wavelengths))

        for selected_indices in combinations(indices, 2 * (self.dimension - 1)):
            intervals = [(selected_indices[i], selected_indices[i + 1]) for i in range(0, len(selected_indices), 2)]

            transition = np.zeros(num_wavelengths)
            for start, end in intervals:
                transition[self.min_transition_size * start:self.min_transition_size * end] = 1

            transitions.append(transition)

        return transitions

    def get_transition(self, index: int) -> npt.NDArray:
        # todo: combinatorial approach instead of this ridiculousness
        return list(self.get_transitions())[index]

    def get_full_colors(self) -> npt.NDArray:
        transitions = self.get_transitions()
        transitions_matrix = np.array(transitions).T

        colors = np.matmul(self.sensor_matrix, transitions_matrix)
        return np.divide(colors, self.whitepoint.reshape((-1, 1)))

