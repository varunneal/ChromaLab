from typing import List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class Spectra:
    def __init__(self, reflectance: Optional[npt.NDArray] = None,
                 wavelengths: Optional[npt.NDArray] = None,
                 data: Optional[npt.NDArray] = None):
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

    def plot(self, name, color):
        plt.plot(self.reflectance[:, 0], self.reflectance[:, 1], label=name, color=color)

    def wavelengths(self) -> npt.NDArray:
        return self.reflectance[:, 0]

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
        # Walowit · 1987
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
