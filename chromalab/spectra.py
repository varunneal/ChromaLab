from typing import List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import warnings

from colour import SDS_ILLUMINANTS, SDS_LIGHT_SOURCES, sd_to_XYZ, XYZ_to_xy, XYZ_to_sRGB, SpectralDistribution, notation


class Spectra:
    def __init__(self, array: Optional[Union[npt.NDArray]] = None,
                 wavelengths: Optional[npt.NDArray] = None, data: Optional[npt.NDArray] = None,
                 normalized: Optional[bool] = True, **kwargs):
        """
        Either provide `reflectance` as a two column NDArray or provide both
        `wavelengths` and `data` as single column NDArrays.
        """
        if array is None:
            array = np.column_stack((wavelengths, data))
        if not isinstance(array, np.ndarray):
            raise TypeError("Input should be a numpy array")

        if array.shape[1] != 2:
            raise ValueError("Array should have two columns")

        first_col = array[:, 0]
        if not (np.all(first_col >= 0)):
            raise ValueError("Wavelengths must be positive.")

        if not np.all(first_col == np.sort(first_col)):
            raise ValueError("Wavelengths should be in ascending order")

        second_col = array[:, 1]
        if normalized and not (np.all(second_col >= 0) and np.all(second_col <= 1)):
            warnings.warn("Reflectance has values not between 0 and 1. Clipping.")
            array[:, 1] = np.clip(array[:, 1], 0, 1)

        self.reflectance = array
        self.normalized = normalized

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    """Converts Spectra to the SpectralDistribution object from Colour library."""
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

    def plot(self, name=None, color=None, ax=None, alpha=1.0):
        if color is None and name is None:
            color = self.to_rgb()
            name = self.__class__.__name__
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
        attrs = self.__dict__.copy()
        attrs["data"] = np.array(interpolated_data)
        return self.__class__(**attrs)

    def interpolated_value(self, wavelength) -> float:
        wavelengths = self.wavelengths()
        idx = np.searchsorted(wavelengths, wavelength)

        if idx == 0:
            return float(self.reflectance[0, 1])
        if idx == len(wavelengths):
            return float(self.reflectance[-1, 1])

        if wavelengths[idx] == wavelength:
            return float(self.reflectance[idx, 1])

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
        # todo: can add wavelength interpolation
        attrs = self.__dict__.copy()
        if isinstance(other, float) or isinstance(other, int):
            attrs["data"] = self.data() + other
        elif isinstance(other, Spectra):
            if not np.array_equal(other.wavelengths(), self.wavelengths()):
                raise ValueError(f"Wavelengths must match for addition.")
            attrs["data"] = self.data() + other.data()
        else:
            raise TypeError("This addition not supported.")

        return self.__class__(**attrs)

    def __mul__(self, scalar: Union[int, float]):
        attrs = self.__dict__.copy()
        attrs["data"] = scalar * self.data()
        return self.__class__(**attrs)

    def __rmul__(self, scalar: Union[int, float]):
        # This method allows scalar * Spectra to work
        return self.__mul__(scalar)

    def __rpow__(self, base: float):
        attrs = self.__dict__.copy()
        attrs["data"] = np.power(base, self.data())
        return self.__class__(**attrs)

    def __pow__(self, exponent: float):
        attrs = self.__dict__.copy()
        attrs["data"] = np.power(self.data(), exponent)
        return self.__class__(**attrs)

    def __truediv__(self, other: Union["Spectra", float, int]):
        attrs = self.__dict__.copy()
        if isinstance(other, (int, float)):
            attrs["data"] = self.data() / other
        elif isinstance(other, Spectra):
            if not np.array_equal(other.wavelengths(), self.wavelengths()):
                raise ValueError("Wavelengths must match for division")
            denom = np.clip(other.data(), 1e-7, None)
            attrs["data"] = self.data() / denom
        return self.__class__(**attrs)

    """Normalize operator, overwriting invert ~ operator."""
    def __invert__(self):
        # Division by maximum element
        attrs = self.__dict__.copy()
        attrs["data"] = self.data() / np.max(self.data())
        attrs["normalized"] = True
        return self.__class__(**attrs)

    def __str__(self):
        # TODO: can be smarter
        return str(self.reflectance)


class Illuminant(Spectra):
    def __init__(self, array: Optional[Union[Spectra, npt.NDArray]] = None,
                 wavelengths: Optional[npt.NDArray] = None, data: Optional[npt.NDArray] = None, **kwargs):
        if isinstance(array, Spectra):
            super().__init__(**array.__dict__, **kwargs)
        else:
            super().__init__(array=array, wavelengths=wavelengths, data=data, **kwargs)

    @staticmethod
    def get(name):
        light = SDS_ILLUMINANTS.get(name)
        if light is None:
            light = SDS_LIGHT_SOURCES.get(name)
        return Illuminant(data=light.values / np.max(light.values), wavelengths=light.wavelengths)
