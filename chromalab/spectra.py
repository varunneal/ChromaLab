from typing import List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import warnings

from colour import SDS_ILLUMINANTS, SDS_LIGHT_SOURCES, sd_to_XYZ, XYZ_to_xy, XYZ_to_sRGB, SpectralDistribution, notation


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
        # if not (np.all(first_col >= 350) and np.all(first_col <= 850)):
        #     raise ValueError("First column values should be between 350 and 850")

        if not np.all(first_col == np.sort(first_col)):
            raise ValueError("First column should be in ascending order")

        second_col = reflectance[:, 1]
        if not (np.all(second_col >= 0) and np.all(second_col <= 1)):
            # TODO: Makes fluorescence imposssible
            # warnings.warn("Reflectance has values not between 0 and 1.")
            raise ValueError("Second column values should be between 0 and 1")

        self.reflectance = reflectance

    def to_colour(self) -> SpectralDistribution:
        return SpectralDistribution(data=self.data(), domain=self.wavelengths())

    def to_xyz(self, illuminant: Optional["Spectra"] = None):
        i = illuminant.to_colour() if illuminant else None

        return sd_to_XYZ(self.to_colour(), illuminant=i) / 100

    def to_rgb(self, illuminant: Optional["Spectra"] = None):
        i = illuminant.to_colour() if illuminant is not None else Illuminant.get("D65").to_colour()
        coord = XYZ_to_xy(sd_to_XYZ(i) / 100)
        return np.clip(XYZ_to_sRGB(self.to_xyz(illuminant), coord), 0, 1)

    def to_hex(self, illuminant: Optional["Spectra"] = None):
        return notation.RGB_to_HEX(np.clip(self.to_rgb(illuminant), 0, 1))

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
            return Spectra(wavelengths=self.wavelengths(), data=self.data() + other)

        if not np.array_equal(other.wavelengths(), self.wavelengths()):
            raise TypeError(f"Interpolation not supported for addition.")
        # todo: can add interpolation

        return Spectra(wavelengths=self.wavelengths(), data=self.data() + other.data())

    def __mul__(self, scalar: Union[int, float]):
        return Spectra(wavelengths=self.wavelengths(), data=scalar * self.data())

    def __rmul__(self, scalar: Union[int, float]):
        # This method allows scalar * Spectra to work
        return self.__mul__(scalar)

    def __pow__(self, exponent: float):
        new_data = np.power(self.data(), exponent)
        return Spectra(wavelengths=self.wavelengths(), data=new_data)

    def __str__(self):
        return str(self.reflectance)


class Illuminant(Spectra):
    def __init__(self, reflectance: Optional[Union[Spectra, npt.NDArray]] = None,
                 wavelengths: Optional[npt.NDArray] = None,
                 data: Optional[npt.NDArray] = None):
        if isinstance(reflectance, Spectra):
            super().__init__(reflectance.reflectance)
        else:
            super().__init__(reflectance=reflectance, wavelengths=wavelengths, data=data)

    @staticmethod
    def get(name):
        light = SDS_ILLUMINANTS.get(name)
        if light is None:
            light = SDS_LIGHT_SOURCES.get(name)
        return Illuminant(data=light.values / np.max(light.values), wavelengths=light.wavelengths)


class Pigment(Spectra):
    def __init__(self, reflectance: Optional[Union[Spectra, npt.NDArray]] = None,
                 k: Optional[npt.NDArray] = None,
                 s: Optional[npt.NDArray] = None,
                 wavelengths: Optional[npt.NDArray] = None):
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
            self.k, self.s = _k, _s

        elif k is not None and s is not None and wavelengths is not None:
            # compute reflectance from k & s
            if not k.shape == s.shape or not k.shape == wavelengths.shape:
                raise ValueError("Coefficients k and s and wavelengths must be same shape.")

            r = 1 + (k / s) - np.sqrt(np.square(k / s) + (2 * k / s))
            super().__init__(wavelengths=wavelengths, data=r)
        else:
            raise ValueError("Must either specify reflectance or k and s coefficients and wavelengths.")

    def compute_k_s(self) -> Tuple[npt.NDArray, npt.NDArray]:
        # Walowit · 1987 specifies this least squares method
        # todo: GJK method as per Centore • 2015
        k, s = [], []
        for wavelength, r in np.clip(self.reflectance, 1e-4, 1):
            # SK Loyalka · 1995 suggests 4 instead of 2. I find 2 is better.
            k_over_s = (1 - r) * (1 - r) / (2 * r)
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
        return self.k, self.s
