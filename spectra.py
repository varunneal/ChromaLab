from typing import List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


# from colour import SDS_ILLUMINANTS, CCS_ILLUMINANTS, convert

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def piecewise_gaussian(x, A, mu, sigma1, sigma2):
    return np.piecewise(x, [x < mu, x >= mu],
                        [lambda t: gaussian(t, A, mu, sigma1), lambda t: gaussian(t, A, mu, sigma2)])


class Spectra:
    def __init__(self, reflectance: Optional[npt.NDArray] = None,
                 wavelengths: Optional[npt.NDArray] = None,
                 data: Optional[npt.NDArray] = None):
        """
        Either provide `reflectance` as a two column NDArray or provide both
        `wavelengths` and `data` as single column NDArrays.
        """
        if reflectance is None:
            reflectance = np.column_stack((wavelengths, data))
        if not isinstance(reflectance, np.ndarray):
            raise TypeError("Input should be a numpy array")

        if reflectance.shape[1] != 2:
            raise ValueError("Array should have two columns")

        first_col = reflectance[:, 0]
        if not (np.all(first_col >= 350) and np.all(first_col <= 850)):
            raise ValueError("First column values should be between 350 and 850")

        if not np.all(first_col == np.sort(first_col)):
            raise ValueError("First column should be in ascending order")

        second_col = reflectance[:, 1]
        if not (np.all(second_col >= 0) and np.all(second_col <= 1)):
            raise ValueError("Second column values should be between 0 and 1")

        self.reflectance = reflectance

        lbda = self.wavelengths()
        x_bar = piecewise_gaussian(lbda, 1.056, 599.8, 37.9, 31.0) + \
            piecewise_gaussian(lbda, 0.362, 442.0, 16.0, 26.7) + \
            piecewise_gaussian(lbda, -0.065, 501.1, 20.4, 26.2)
        y_bar = piecewise_gaussian(lbda, 0.821, 568.8, 46.9, 40.5) + \
                piecewise_gaussian(lbda, 0.286, 530.9, 16.3, 31.1)
        z_bar = piecewise_gaussian(lbda, 1.217, 437.0, 11.8, 36.0) + \
                piecewise_gaussian(lbda, 0.681, 459.0, 26.0, 13.8)

        self.xyz_matrix = np.stack([x_bar, y_bar, z_bar])

    def to_xyz(self):
        # some problems:
        # xyz matrix is generated assuming a specific whitepoint
        # the data is generated using a (perhaps distinct) specific whitepoint

        # see https://en.wikipedia.org/wiki/CIE_1931_color_space#Color_matching_functions
        xyz = np.matmul(self.xyz_matrix, self.data())
        whitepoint = np.matmul(self.xyz_matrix, np.ones_like(self.data()))

        return np.divide(xyz, whitepoint)

    def to_rgb(self):
        # todo: this is bad for some reason. Have tried multiple matrices.
        # see https://en.wikipedia.org/wiki/CIE_1931_color_space#Construction_of_the_CIE_XYZ_color_space_from_the_Wright%E2%80%93Guild_data
        # and http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
        xyz = self.to_xyz()
        # transform_matrix = np.array([
        #     [2.37, -0.9, -0.47],
        #     [-0.51, 1.4, 0.09],
        #     [0.005, -0.01, 1.01]
        # ])
        transform_matrix = np.array([
            [2.85, -1.36, -0.47],
            [-1.09, 2.03, 0.227],
            [0.103, -0.296, 1.45]
        ])
        rgb = np.matmul(transform_matrix, xyz)
        return np.clip(rgb, 0, 1)

    def plot(self, name="spectra", color=None, ax=None, alpha=1.0):
        if color is None:
            color = self.to_rgb()
        if not ax:
            plt.plot(self.reflectance[:, 0], self.reflectance[:, 1], label=name, color=color, alpha=alpha)
        else:
            ax.plot(self.reflectance[:, 0], self.reflectance[:, 1], label=name, color=color, alpha=alpha)

    def interpolate_values(self, wavelengths: Union[npt.NDArray, None]) -> 'Spectra':
        if wavelengths is None:
            return self
        interpolated_data = []
        for wavelength in wavelengths:
            d = self.interpolated_value(wavelength)
            interpolated_data.append(d)
        return Spectra(wavelengths=wavelengths, data=np.array(interpolated_data))


    def interpolated_value(self, wavelength) -> float:
        wavelengths = self.wavelengths()
        idx = np.searchsorted(wavelengths, wavelength)

        if idx == 0 or idx == len(wavelengths):
            return 0

        if wavelengths[idx] == wavelength:
            return self.reflectance[idx, 1]

        # Linearly interpolate between the nearest wavelengths
        x1, y1 = wavelengths[idx - 1], self.reflectance[idx - 1, 1]
        x2, y2 = wavelengths[idx], self.reflectance[idx, 1]

        return y1 + (y2 - y1) * (wavelength - x1) / (x2 - x1)

    def wavelengths(self) -> npt.NDArray:
        return self.reflectance[:, 0]

    def data(self) -> npt.NDArray:
        return self.reflectance[:, 1]

    def __add__(self, other: Union['Spectra', float, int]) -> 'Spectra':
        # todo: can add ndarray support
        if isinstance(other, float) or isinstance(other, int):
            return Spectra(wavelengths=self.wavelengths(), data=np.clip(self.data() + other, 0, 1))

        if not np.array_equal(other.wavelengths(), self.wavelengths()):
            raise TypeError(f"Interpolation not supported for addition.")
        # todo: can add interpolation

        return Spectra(wavelengths=self.wavelengths(), data=self.data() + other.data())

    def __mul__(self, scalar: Union[int, float]):
        return Spectra(wavelengths=self.wavelengths(), data=np.clip(scalar * self.data(), 0, 1))

    def __rmul__(self, scalar: Union[int, float]):
        # This method allows scalar * Spectra to work
        return self.__mul__(scalar)

    def __pow__(self, exponent: float):
        new_data = np.power(self.data(), exponent)
        return Spectra(wavelengths=self.wavelengths(), data=new_data)

    def __str__(self):
        return str(self.reflectance)


class Pigment(Spectra):
    def __init__(self, reflectance: Optional[Union[Spectra, npt.NDArray]] = None,
                 k: Optional[npt.NDArray] = None,
                 s: Optional[npt.NDArray] = None):
        """
        Either pass in @param reflectance or pass in
        @param k and @param s.

        k and s are stored as spectra rather than NDArrays.
        """
        if reflectance is not None:
            # compute k & s from reflectance
            if isinstance(reflectance, Spectra):
                super().__init__(reflectance.reflectance)
            else:
                super().__init__(reflectance)
            _k, _s = self.compute_k_s()
            wavelengths = self.reflectance[:, 0]
            self.k, self.s = Spectra(wavelengths=wavelengths, data=_k), Spectra(wavelengths=wavelengths, data=_s)

        elif k is not None and s is not None:
            # compute reflectance from k & s
            if not k.shape == s.shape:
                raise ValueError("Coefficients k and s must be same shape.")
            self.k = Spectra(k)
            self.s = Spectra(s)
            _k, _s = k[:, 1], s[:, 1]
            r = 1 + (_k / _s) - np.sqrt(np.square(_k / _s) + (2 * _k / _s))
            super().__init__(wavelengths=self.k.wavelengths(), data=r)
        else:
            raise ValueError("Must either specify reflectance or k and s coefficients.")

    def compute_k_s(self) -> Tuple[npt.NDArray, npt.NDArray]:
        # Walowit · 1987 specifies this least squares method
        # todo: GJK method as per Centore • 2015
        k, s = [], []
        for wavelength, r in self.reflectance:
            # SK Loyalka · 1995 for corrected coefficient
            k_over_s = (1 - r) * (1 - r) / (4 * r)
            A = np.array([[-1, k_over_s], [1, 1]])
            b = np.array([0, 1])

            AtA_inv = np.linalg.inv(np.dot(A.T, A))
            Atb = np.dot(A.T, b)

            _k, _s = np.clip(np.dot(AtA_inv, Atb), 0, 1)
            k.append(_k)
            s.append(_s)

        return np.array(k), np.array(s)

    def get_k_s(self) -> Tuple[npt.NDArray, npt.NDArray]:
        # todo: pass in wavelength list for interpolation/sampling consistency with mixing
        return self.k.reflectance[:, 1], self.s.reflectance[:, 1]
