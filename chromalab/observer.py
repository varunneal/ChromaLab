from itertools import combinations
from typing import List, Union, Optional

import numpy as np
import numpy.typing as npt

from .spectra import Spectra


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def remove_trailing_nans(arr):
    mask = np.any(np.isnan(arr), axis=1)
    idx = np.where(mask)[0]
    if idx.size > 0:
        last_valid_idx = np.where(~mask)[0][-1]
        return arr[:last_valid_idx + 1]
    return arr


def neitz_cone(wavelengths, lambda_max=559, OD=0.30, output='alog'):
    wavelengths = wavelengths.astype(np.float32)

    A = 0.417050601
    B = 0.002072146
    C = 0.000163888
    D = -1.922880605
    E = -16.05774461
    F = 0.001575426
    G = 5.11376e-05
    H = 0.00157981
    I = 6.58428e-05
    J = 6.68402e-05
    K = 0.002310442
    L = 7.31313e-05
    M = 1.86269e-05
    N = 0.002008124
    O = 5.40717e-05
    P = 5.14736e-06
    Q = 0.001455413
    R = 4.217640000e-05
    S = 4.800000000e-06
    T = 0.001809022
    U = 3.86677000e-05
    V = 2.99000000e-05
    W = 0.001757315
    X = 1.47344000e-05
    Y = 1.51000000e-05
    Z = OD + 0.00000001

    A2 = (np.log10(1.00000000 / lambda_max) - np.log10(1.00000000 / 558.5))
    vector = np.log10(np.reciprocal(wavelengths))
    const = 1 / np.sqrt(2 * np.pi)

    ex_temp1 = np.log10(-E + E * np.tanh(-((10 ** (vector - A2)) - F) / G)) + D
    ex_temp2 = A * np.tanh(-(((10 ** (vector - A2))) - B) / C)
    ex_temp3 = -(J / I * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - H) / I) ** 2)))
    ex_temp4 = -(M / L * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - K) / L) ** 2)))
    ex_temp5 = -(P / O * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - N) / O) ** 2)))
    ex_temp6 = (S / R * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - Q) / R) ** 2)))
    ex_temp7 = ((V / U * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - T) / U) ** 2))) / 10)
    ex_temp8 = ((Y / X * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - W) / X) ** 2))) / 100)
    ex_temp = ex_temp1 + ex_temp2 + ex_temp3 + ex_temp4 + ex_temp5 + ex_temp6 + ex_temp7 + ex_temp8

    OD_temp = np.log10((1 - 10 ** -((10 ** ex_temp) * Z)) / (1 - 10 ** -Z))

    if output == 'log':
        extinction = ex_temp
        with_OD = OD_temp
    else:
        extinction = 10 ** ex_temp
        with_OD = 10 ** OD_temp

    return with_OD, extinction


class Cone(Spectra):
    cone_data = np.genfromtxt('../data/cones/ss2deg_10lin.csv')

    def __init__(self, reflectance: Optional[Union[Spectra, npt.NDArray]] = None,
                 wavelengths: Optional[npt.NDArray] = None,
                 data: Optional[npt.NDArray] = None):
        if isinstance(reflectance, Spectra):
            super().__init__(reflectance.reflectance)
        else:
            super().__init__(reflectance=reflectance, wavelengths=wavelengths, data=data)

    def observe(self, spectra: Spectra, illuminant: Spectra):
        return np.divide(np.dot(self.data(), spectra.data()), np.dot(self.data(), illuminant.data()))

    @staticmethod
    def neitz_l(wavelengths=None):
        if wavelengths is None:
            wavelengths = np.arange(390, 701, 1)

        data = np.clip(neitz_cone(wavelengths, lambda_max=559, OD=0.35, output="alog")[1], 0, 1)
        return Cone(wavelengths=wavelengths, data=data)

    @staticmethod
    def neitz_q(wavelengths=None):
        if wavelengths is None:
            wavelengths = np.arange(390, 701, 1)

        data = np.clip(neitz_cone(wavelengths, lambda_max=545, OD=0.285, output="alog")[1], 0, 1)
        return Cone(wavelengths=wavelengths, data=data)


    @staticmethod
    def neitz_m(wavelengths=None):
        if wavelengths is None:
            wavelengths = np.arange(390, 701, 1)

        data = np.clip(neitz_cone(wavelengths, lambda_max=530, OD=0.22, output="alog")[1], 0, 1)
        return Cone(wavelengths=wavelengths, data=data)


    @staticmethod
    def neitz_s(wavelengths=None):
        if wavelengths is None:
            wavelengths = np.arange(390, 701, 1)

        data = np.clip(neitz_cone(wavelengths, lambda_max=419, OD=0.35, output="alog")[1], 0, 1)
        return Cone(wavelengths=wavelengths, data=data)


    @staticmethod
    def l_cone(wavelengths=None):
        # 390 to 700
        reflectances = Spectra(Cone.cone_data[:311, [0, 1]])
        return Cone(reflectances.interpolate_values(wavelengths))

    @staticmethod
    def shift_cone(shift, wavelengths=None):
        m_cone = Cone.m_cone(wavelengths)
        r = [(w, 1e-4) for w in m_cone.wavelengths() if w < m_cone.wavelengths()[0] + shift] + \
            [(w + shift, v) for (w, v) in m_cone.reflectance if w + shift <= m_cone.wavelengths()[-1]]
        return Cone(reflectance=np.array(r))

    @staticmethod
    def q_cone(wavelengths=None):
        return Cone.shift_cone(15, wavelengths=wavelengths)

    @staticmethod
    def m_cone(wavelengths=None):
        # 390 to 700
        reflectances = Spectra(Cone.cone_data[:311, [0, 2]])
        return Cone(reflectances.interpolate_values(wavelengths))

    @staticmethod
    def s_cone(wavelengths=None):
        # 390 to 700
        reflectances = Spectra(remove_trailing_nans(Cone.cone_data[:311, [0, 3]]))
        return Cone(reflectances.interpolate_values(wavelengths))



def get_m_transitions(m, wavelengths, both_types=True):
    all_transitions = []
    for combination in combinations(range(len(wavelengths)), m):
        arr = np.zeros_like(wavelengths, dtype=int)
        for index in combination:
            arr[index:] = 1 - arr[index]
        all_transitions.append(arr)
        if both_types:
            all_transitions.append(1-arr)

    return all_transitions


class Observer:
    def __init__(self, sensors: List[Spectra], illuminant: Optional[Spectra] = None, min_transition_size: int = 10):
        self.dimension = len(sensors)
        self.sensors = sensors

        total_wavelengths = np.unique(np.concatenate([sensor.wavelengths() for sensor in sensors]))
        self.wavelengths = total_wavelengths

        self.sensor_matrix = self.get_sensor_matrix(total_wavelengths)
        if illuminant is not None:
            illuminant = illuminant.interpolate_values(self.wavelengths).data()
        else:
            illuminant = np.ones_like(self.wavelengths)

        self.illuminant = illuminant
        self.min_transition_size = min_transition_size

    @staticmethod
    def trichromat(wavelengths=None, illuminant=None):
        l_cone = Cone.l_cone(wavelengths)
        m_cone = Cone.m_cone(wavelengths)
        s_cone = Cone.s_cone(wavelengths)
        return Observer([s_cone, m_cone, l_cone], illuminant=illuminant)

    @staticmethod
    def neitz_trichromat(wavelengths=None, illuminant=None):
        l_cone = Cone.neitz_l(wavelengths)
        m_cone = Cone.neitz_m(wavelengths)
        s_cone = Cone.neitz_s(wavelengths)
        return Observer([s_cone, m_cone, l_cone], illuminant=illuminant)

    @staticmethod
    def neitz_tetrachromat(wavelengths=None, illuminant=None):
        l_cone = Cone.neitz_l(wavelengths)
        q_cone = Cone.neitz_q(wavelengths)
        m_cone = Cone.neitz_m(wavelengths)
        s_cone = Cone.neitz_s(wavelengths)
        return Observer([s_cone, m_cone, q_cone, l_cone], illuminant=illuminant)

    @staticmethod
    def tetrachromat(wavelengths=None, illuminant=None):
        l_cone = Cone.l_cone(wavelengths)
        q_cone = Cone.q_cone(wavelengths)
        m_cone = Cone.m_cone(wavelengths)
        s_cone = Cone.s_cone(wavelengths)
        return Observer([s_cone, m_cone, q_cone, l_cone], illuminant=illuminant)

    def get_whitepoint(self, wavelengths: Optional[npt.NDArray] = None):
        sensor_matrix = self.get_sensor_matrix(wavelengths)

        return np.matmul(sensor_matrix, self.illuminant)

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

        observed_color = np.matmul(self.sensor_matrix, data * self.illuminant)
        whitepoint = np.matmul(self.sensor_matrix, self.illuminant)

        return np.divide(observed_color, whitepoint)

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
