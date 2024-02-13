from psychopy.hardware import findPhotometer
import numpy as np

import sys

name = sys.argv[1]
photom = findPhotometer(device='PR650')
w, d = photom.getSpectrum()
np.save(f"pr650_data/{name}_wavelengths", w)
np.save(f"pr650_data/{name}_data", d)

