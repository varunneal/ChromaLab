from itertools import combinations
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt


from .observer import getsRGBfromWavelength


class MaxDisplayBasis:
    dim4SampleConst = 10
    dim3SampleConst = 5
    
    def __init__(self, observer, verbose=False) -> None:
        self.verbose = verbose
        self.observer = observer
        self.wavelengths = observer.wavelengths
        self.matrix = observer.get_normalized_sensor_matrix()
        self.dimension = observer.dimension

        self.max_primaries = self.computeMaxPrimaries()
    

    def computeMaxPrimaries(self):

        def computeVolume(wavelengths):
            # set of n wavelengths, and we want to pick the monochromatic wavelength in each direction
            idxs = np.searchsorted(self.wavelengths, wavelengths) # pick index wavelengths from list of wavelengths
            submat = self.matrix[:, idxs] # list of basis vectors per index in cone space
            vol = np.linalg.det(submat) # volume of resulting parallelepiped -> according to chatGPT, it is divided by a fixed 1/n! factor based on dimensionality, so doesn't matter if we divide or not. 
            return vol

        data = list(combinations(self.wavelengths, self.dimension))
        # with Pool(processes=8) as pool:
        #     result = pool.imap_unordered(computeVolume, data)
        #     maxvol = reduce(max, result)
        
        # result = [ (i, computeVolume(x)) for i, x in enumerate(data)]
        # idx, maxvol = reduce(lambda x, y: max(x[1], y[]), result)

        result = list(map(computeVolume, data))
        idx = np.argmax(result)
        max_primaries = list(data)[idx]
        print(max_primaries)
        return max_primaries
    
    def displayMaxPrimaries(self):

        color_patches = [getsRGBfromWavelength(x) for x in self.max_primaries]
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the color patches
        for i, color in enumerate(color_patches):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.clip(color, 0, 1)))

        # Set the x-axis limits
        ax.set_xlim(0, len(color_patches))

        # Remove the y-axis ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])

        # Show the plot
        plt.show()