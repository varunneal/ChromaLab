from itertools import combinations
from typing import List, Union, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from importlib import resources

from .spectra import Spectra


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def BaylorNomogram(wls, lambdaMax: int):
    """
    Baylor, Nunn, and Schnapf, 1987.
    """
    # These are the coefficients for the polynomial approximation.
    aN = np.array([-5.2734, -87.403, 1228.4, -3346.3, -5070.3, 30881, -31607])

    wlsum = wls / 1000.0
    wlsVec = np.log10((1.0 / wlsum) * lambdaMax / 561)
    logS = aN[0] + aN[1] * wlsVec + aN[2] * wlsVec ** 2 + aN[3] * wlsVec ** 3 + \
           aN[4] * wlsVec ** 4 + aN[5] * wlsVec ** 5 + aN[6] * wlsVec ** 6
    T = 10 ** logS
    return Cone(data=T.T, wavelengths=wls, quantal=True)


def GovardovskiiNomogram(S, lambdaMax):
    """
    Victor I. Govardovskii et al., 2000,
    """
    # Valid range of wavelength for A1-based visual pigments
    Lmin, Lmax = 330, 700

    # Valid range of lambdaMax value
    lmaxLow, lmaxHigh = 350, 600

    # Alpha-band parameters
    A, B, C = 69.7, 28, -14.9
    D = 0.674
    b, c = 0.922, 1.104

    # Beta-band parameters
    Abeta = 0.26

    # Assuming S is directly the wavelengths array
    wls = np.array(S)

    nWls = len(wls)
    T_absorbance = np.zeros((1, nWls))  # nT is assumed to be 1 based on user note

    if lmaxLow < lambdaMax < lmaxHigh:
        # alpha-band polynomial
        a = 0.8795 + 0.0459 * np.exp(-(lambdaMax - 300) ** 2 / 11940)

        x = lambdaMax / wls
        midStep1 = np.exp(np.array([A, B, C]) * np.array([a, b, c]) - x[:, None] * np.array([A, B, C]))
        midStep2 = np.sum(midStep1, axis=1) + D

        S_x = 1 / midStep2

        # Beta-band polynomial
        bbeta = -40.5 + 0.195 * lambdaMax
        lambdaMaxbeta = 189 + 0.315 * lambdaMax

        midStep1 = -((wls - lambdaMaxbeta) / bbeta) ** 2
        S_beta = Abeta * np.exp(midStep1)

        # alpha band and beta band together
        T_absorbance[0, :] = S_x + S_beta

        # Zero sensitivity outside valid range
        T_absorbance[0, wls < Lmin] = 0
        T_absorbance[0, wls > Lmax] = 0
    else:
        raise ValueError(f'Lambda Max {lambdaMax} not in range of nomogram')

    return Cone(data=np.clip(T_absorbance.T, 0, 1), wavelengths=wls, quantal=True)


class Cone(Spectra):
    # TODO: refactor after analysis
    with resources.path("chromalab.cones", "ss2deg_10lin.csv") as data_path:
        cone_data = pd.read_csv(data_path, header=None).iloc[:-130, :]

    def __init__(self, array: Optional[Union[Spectra, npt.NDArray]] = None,
                 wavelengths: Optional[npt.NDArray] = None, data: Optional[npt.NDArray] = None,
                 quantal=False, **kwargs):
        self.quantal = quantal
        if isinstance(array, Spectra):
            super().__init__(**array.__dict__, **kwargs)
        else:
            super().__init__(array=array, wavelengths=wavelengths, data=data, **kwargs)

    def observe(self, spectra: Spectra, illuminant: Spectra):
        return np.divide(np.dot(self.data(), spectra.data()), np.dot(self.data(), illuminant.data()))

    def as_quantal(self):
        if self.quantal:
            return self
        log_data = np.log(self.data()) - np.log(self.wavelengths())
        quantal_data = np.exp(log_data - np.max(log_data))
        attrs = self.__dict__.copy()
        attrs["data"] = quantal_data
        attrs["quantal"] = True
        return self.__class__(**attrs)

    def as_energy(self):
        if not self.quantal:
            return self
        log_data = np.log(self.wavelengths()) + np.log(self.data())
        energy_data = np.exp(log_data - np.max(log_data))
        attrs = self.__dict__.copy()
        attrs["data"] = energy_data
        attrs["quantal"] = False
        return self.__class__(**attrs)

    def with_od(self, od):
        if not self.quantal:
            return self.as_quantal().with_od(od).as_energy()
        od_data = np.divide(1 - np.exp(np.log(10) * -od * self.data()), 1 - (10 ** -od))
        attrs = self.__dict__.copy()
        attrs["data"] = od_data
        return self.__class__(**attrs)

    @staticmethod
    def l_cone(wavelengths=None):
        reflectances = Cone.cone_data.iloc[:, [0,1]].to_numpy()
        return Cone(reflectances).interpolate_values(wavelengths)

    @staticmethod
    def shift_cone(shift, wavelengths=None):
        # todo: i think this is only works with 1nm stepsize wavelengths lmao
        l_cone = Cone.l_cone()
        r = [(w, 1e-4) for w in l_cone.wavelengths() if w < l_cone.wavelengths()[0] + shift] + \
            [(w + shift, v) for (w, v) in l_cone.reflectance if w + shift <= l_cone.wavelengths()[-1]]
        return Cone(array=np.array(r)).interpolate_values(wavelengths)

    @staticmethod
    def q_cone(wavelengths=None):
        return Cone.shift_cone(-15, wavelengths=wavelengths)

    @staticmethod
    def m_cone(wavelengths=None):
        reflectances = Cone.cone_data.iloc[:, [0,2]].to_numpy()
        return Cone(reflectances).interpolate_values(wavelengths)

    @staticmethod
    def s_cone(wavelengths=None):
        reflectances = Cone.cone_data.iloc[:, [0,3]].dropna().to_numpy()
        return Cone(reflectances).interpolate_values(wavelengths)


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
