from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

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


def main():
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

        mixed_pigment.plot(name="", color=interp_color_str, alpha=alpha_value)

    yellow.plot(name='yellow', color='yellow')
    magenta.plot(name='magenta', color='magenta')
    plt.legend()
    plt.show()


main()
