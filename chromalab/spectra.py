from typing import List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import warnings
import copy

from colour import SDS_ILLUMINANTS, SDS_LIGHT_SOURCES, sd_to_XYZ, XYZ_to_xy, XYZ_to_sRGB, SpectralDistribution, notation


class Spectra:
    def __init__(self, array: Optional[Union[npt.NDArray]] = None,
                 wavelengths: Optional[npt.NDArray] = None, data: Optional[npt.NDArray] = None,
                 normalized: Optional[bool] = True, **kwargs):
        """
        Either provide `reflectance` as a two column NDArray or provide both
        `wavelengths` and `data` as single column NDArrays.
        """
        if array is not None:
            if not isinstance(array, np.ndarray):
                raise TypeError("Input should be a numpy array")
            if array.shape[1] != 2:
                raise ValueError("Array should have two columns")
            wavelengths = array[:, 0]
            data = array[:, 1]

        if not (np.all(wavelengths >= 0)):
            raise ValueError("Wavelengths must be positive.")

        if not np.all(wavelengths == np.sort(wavelengths)):
            raise ValueError("Wavelengths should be in ascending order")

        if normalized and not (np.all(data >= 0) and np.all(data <= 1)):
            warnings.warn("Data has values not between 0 and 1. Clipping.")
            data = np.clip(data, 0, 1)

        self.wavelengths = wavelengths.reshape(-1)
        self.data = data.reshape(-1)
        self.normalized = normalized

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def array(self) -> npt.NDArray:
        return np.column_stack((self.wavelengths, self.data))

    @staticmethod
    def from_transitions(transitions, start, wavelengths, maxVal = 1) -> 'Spectra':
        step = wavelengths[1] - wavelengths[0]
        minwave = wavelengths[0]
        maxwave = wavelengths[-1] + step

        transitions = copy.deepcopy(transitions)
        transitions.insert(0, minwave)
        transitions.insert(len(transitions), maxwave)
        transitions = [round(t, 2) for t in transitions]
        ref = []
        for i in range(len(transitions)-1):
            ref += [np.full(int(round((transitions[i+1] - transitions[i])/ step)), (start + i) % 2)]
        ref = np.concatenate(ref)
        assert(len(ref) == len(wavelengths))
        data = ref * maxVal
        return Spectra(wavelengths=wavelengths, data=data)

    """Converts Spectra to the SpectralDistribution object from Colour library."""
    def to_colour(self) -> SpectralDistribution:
        return SpectralDistribution(data=self.data, domain=self.wavelengths)

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
            # name = self.__class__.__name__
        if not ax:
            plt.plot(self.wavelengths, self.data, label=name, color=color, alpha=alpha)
        else:
            ax.plot(self.wavelengths, self.data, label=name, color=color, alpha=alpha)

    def interpolate_values(self, wavelengths: Union[npt.NDArray, None]) -> 'Spectra':
        if wavelengths is None:
            return self
        interpolated_data = []
        for wavelength in wavelengths:
            d = self.interpolated_value(wavelength)
            interpolated_data.append(d)
        attrs = self.__dict__.copy()

        attrs["data"] = np.array(interpolated_data)
        attrs["wavelengths"] = wavelengths
        return self.__class__(**attrs)

    def interpolated_value(self, wavelength) -> float:
        idx = np.searchsorted(self.wavelengths, wavelength)

        if idx == 0:
            return float(self.data[0])
        if idx == len(self.wavelengths):
            return float(self.data[-1])

        if self.wavelengths[idx] == wavelength:
            return float(self.data[idx])

        # Linearly interpolate between the nearest wavelengths
        x1, y1 = self.wavelengths[idx - 1], self.data[idx - 1]
        x2, y2 = self.wavelengths[idx], self.data[idx]

        return y1 + (y2 - y1) * (wavelength - x1) / (x2 - x1)

    def __getitem__(self, wavelength):
        return self.interpolated_value(wavelength)

    def __add__(self, other: Union['Spectra', float, int]) -> 'Spectra':
        # todo: can add ndarray support
        # todo: can add wavelength interpolation
        attrs = self.__dict__.copy()

        if isinstance(other, float) or isinstance(other, int):
            attrs["data"] = self.data + other
        elif isinstance(other, Spectra):
            if not np.array_equal(other.wavelengths, self.wavelengths):
                raise ValueError(f"Wavelengths must match for addition.")
            attrs["data"] = self.data + other.data
        else:
            raise TypeError("This addition not supported.")

        return self.__class__(**attrs)

    def __rsub__(self, other: Union[float, int]) -> 'Spectra':
        attrs = self.__dict__.copy()

        if isinstance(other, (float, int)):
            attrs["data"] = other - self.data
        else:
            raise TypeError("This subtraction not supported from the left side with a non-numeric type.")

        return self.__class__(**attrs)

    def __mul__(self, scalar: Union[int, float]):
        attrs = self.__dict__.copy()
        attrs["data"] = scalar * self.data
        return self.__class__(**attrs)

    def __rmul__(self, scalar: Union[int, float]):
        # This method allows scalar * Spectra to work
        return self.__mul__(scalar)

    def __rpow__(self, base: float):
        attrs = self.__dict__.copy()
        attrs["data"] = np.power(base, self.data)
        return self.__class__(**attrs)

    def __pow__(self, exponent: float):
        attrs = self.__dict__.copy()
        attrs["data"] = np.power(self.data, exponent)
        return self.__class__(**attrs)

    def __truediv__(self, other: Union["Spectra", float, int]):
        attrs = self.__dict__.copy()
        if isinstance(other, (int, float)):
            attrs["data"] = self.data / other
        elif isinstance(other, Spectra):
            if not np.array_equal(other.wavelengths, self.wavelengths):
                raise ValueError("Wavelengths must match for division")
            denom = np.clip(other.data, 1e-7, None)
            attrs["data"] = self.data / denom
        return self.__class__(**attrs)

    """Normalize operator, overwriting invert ~ operator."""
    def __invert__(self):
        # Division by maximum element
        attrs = self.__dict__.copy()
        attrs["data"] = self.data / np.max(self.data)
        attrs["normalized"] = True
        return self.__class__(**attrs)

    def __str__(self):
        # TODO: can be smarter
        return str(self.wavelengths)


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

def convert_refs_to_spectras(refs, wavelengths) -> List[Spectra]:
    refs = [np.concatenate([wavelengths[:, np.newaxis], ref[:, np.newaxis]], axis=1) for ref in refs]
    return [Spectra(ref) for ref in refs]