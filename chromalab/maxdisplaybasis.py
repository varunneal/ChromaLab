from itertools import combinations
from multiprocess import Pool
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt


from .observer import getsRGBfromWavelength, transformToChromaticity
from scipy.spatial import ConvexHull, convex_hull_plot_2d


class MaxDisplayBasis:
    dim4SampleConst = 10
    dim3SampleConst = 5
    
    def __init__(self, observer, verbose=False) -> None:
        self.verbose = verbose
        self.observer = observer
        self.wavelengths = observer.wavelengths
        self.matrix = observer.get_normalized_sensor_matrix()
        self.chromaticity_mat = transformToChromaticity(self.matrix.T).T
        self.dimension = observer.dimension

        self.max_primaries = self.computeMaxPrimariesInChromaticity() # computeMaxPrimaries()
    

    def computeMaxPrimaries(self, isCPU=False):

        def computeVolume(wavelengths):
            # set of n wavelengths, and we want to pick the monochromatic wavelength in each direction
            idxs = np.searchsorted(self.wavelengths, wavelengths) # pick index wavelengths from list of wavelengths
            submat = self.matrix[:, idxs] # list of basis vectors per index in cone space
            vol = np.linalg.det(submat) # volume of resulting parallelepiped -> according to chatGPT, it is divided by a fixed 1/n! factor based on dimensionality, so doesn't matter if we divide or not. 
            return vol

        data = list(combinations(self.wavelengths, self.dimension))
        # with Pool(processes=1) as pl:
        #     result = pl.map(computeVolume, data)
        result = list(map(computeVolume, data))
        idx = np.argmax(result)
        max_primaries = list(data)[idx]
        print(f"Max Primaries -- {max_primaries}")
        print(f"Max Volume = {result[idx]}")
        return max_primaries
    
    def computeMaxPrimariesInChromaticity(self):

        def computeVolume(wavelengths):
            # set of n wavelengths, and we want to pick the monochromatic wavelength in each direction
            idxs = np.searchsorted(self.wavelengths, wavelengths) # pick index wavelengths from list of wavelengths
            matrix = np.ones((self.dimension, self.dimension))
            matrix[:self.dimension-1, :] = self.chromaticity_mat[:, idxs] # list of basis vectors per index in cone space
            vol = np.linalg.det(matrix) # volume of resulting parallelepiped -> according to chatGPT, it is divided by a fixed 1/n! factor based on dimensionality, so doesn't matter if we divide or not. 
            return vol

        data = list(combinations(self.wavelengths, self.dimension))
        # with Pool(processes=1) as pl:
        #     result = pl.map(computeVolume, data)
        result = list(map(computeVolume, data))
        idx = np.argmax(result)
        max_primaries = list(data)[idx]
        print(f"Max Primaries -- {max_primaries}")
        print(f"Max Volume = {result[idx]}")
        return max_primaries
    
    def computeProjectedConvexHull(self):
        hull = ConvexHull(self.matrix.T)
        points = transformToChromaticity(hull.points)
        hull = ConvexHull(points)
        return points, hull
    
    def displayMaxPrimariesOnChromaticity(self, primaries=None):
        
        def plot_2d_hull(points, hull, ax, color, opacity):
            from matplotlib.collections import LineCollection
            from matplotlib.patches import Polygon

            line_segments = [hull.points[simplex] for simplex in hull.simplices]
            ax.add_collection(LineCollection(line_segments,
                                            colors='k',
                                            linestyle='solid'))
            cent = np.mean(points, axis=0)
            pts = points[hull.vertices]

            k = 1.0
            poly = Polygon(k*(pts - cent) + cent, closed=True,
                        capstyle='round', facecolor=color, alpha=opacity)
            plt.gca().add_patch(poly)

        fig = plt.figure(figsize=plt.figaspect(0.33))

        max_primaries = self.max_primaries if primaries is None else primaries
        color_patches = [getsRGBfromWavelength(x) for x in max_primaries]

        ax = fig.add_subplot(1, 3, 1)
        ax.set_title("Monochromatic Wavelengths")
        ax.set_xlim(self.wavelengths[0], self.wavelengths[-1])
        colors = ['b', 'g', 'r'] if self.dimension == 3 else ['b', 'g', 'y', 'r']
        for i, x in enumerate(max_primaries):
            ax.axvline(x, color=np.clip(color_patches[i], 0, 1))
            ax.plot(self.wavelengths, self.observer.get_sensor_matrix()[i], c=colors[i], alpha=0.5)

        matrix = self.chromaticity_mat if primaries is None else primaries
        if self.dimension <= 3:
            ax = fig.add_subplot(1, 3, 2)
            ax.set_title(f"Ideal Hull with {self.dimension} primaries")
            ax.plot(matrix[0], matrix[1])
            idxs = np.searchsorted(self.wavelengths, max_primaries)
            points = self.chromaticity_mat[:, idxs].T
            hull1 = ConvexHull(points)
            # convex_hull_plot_2d(hull, ax)
            plot_2d_hull(points, hull1, ax, 'k', 0.2)
            ax.scatter(points[:, 0], points[:, 1], c=colors, s=100)

            # Ideal Hull 
            ax = fig.add_subplot(1, 3, 3)
            ax.set_title(f"Maximum Possible Display Gamut")
            ax.plot(matrix[0], matrix[1])
            points, hull2 = self.computeProjectedConvexHull()
            plot_2d_hull(points, hull2, ax, 'k', 0.2)
            ax.scatter(points[:, 0], points[:, 1], c='orange', s=100)

            print(f"Volume Ratio Between n primaries / ideal = {hull1.volume/hull2.volume}")

        else:
            
            ax1 = fig.add_subplot(1, 3, 2, projection='3d')
            ax1.set_title(f"Ideal Hull with {self.dimension} primaries")
            ax1.plot(matrix[0], matrix[1], matrix[2])

            idxs = np.searchsorted(self.wavelengths, max_primaries)
            
            chosen_primaries = np.array([matrix[:, i] for i in idxs])
            chosen_primaries = np.vstack([chosen_primaries, np.zeros(3)])
            for i, primary in enumerate(chosen_primaries):
                if i == len(chosen_primaries)-1: 
                    continue
                ax1.plot([0, primary[0]], [0, primary[1]], [0, primary[2]], color=np.clip(color_patches[i], 0, 1))

            # Compute & plot the convex hull of the matrix
            hull1 = ConvexHull(chosen_primaries)
            ax1.plot_trisurf(chosen_primaries[:, 0], chosen_primaries[:, 1], chosen_primaries[:, 2], triangles=hull1.simplices, color='gray', alpha=0.2)

            # Ideal Hull
            ax2 = fig.add_subplot(1, 3, 3, projection='3d')
            ax2.set_title(f"Maximum Possible Display Gamut")
            ax2.plot(matrix[0], matrix[1], matrix[2])

            _, hull2 = self.computeProjectedConvexHull()
            ax2.plot_trisurf(hull2.points[:, 0], hull2.points[:, 1], hull2.points[:, 2], triangles=hull2.simplices, color='gray', alpha=0.2)
            ax1.shareview(ax2)
            print(f"Volume Ratio Between n primaries / ideal = {hull1.volume/hull2.volume}")

        plt.show()

    def displayMaxPrimaries(self):
        fig = plt.figure(figsize=plt.figaspect(0.5))

        color_patches = [getsRGBfromWavelength(x) for x in self.max_primaries]

        ax = fig.add_subplot(1, 2, 1)

        ax.set_xlim(self.wavelengths[0], self.wavelengths[-1])
        colors = ['b', 'g', 'r'] if self.dimension == 3 else ['b', 'g', 'y', 'r']
        for i, x in enumerate(self.max_primaries):
            ax.axvline(x, color=np.clip(color_patches[i], 0, 1))
            ax.plot(self.wavelengths, self.observer.get_sensor_matrix()[i], c=colors[i], alpha=0.5)

        matrix = self.matrix if self.dimension == 3 else transformToChromaticity(self.matrix.T).T
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot(matrix[0], matrix[1], matrix[2])
        ax.set_xlabel('S')
        ax.set_ylabel('M')
        ax.set_zlabel('L')
        
        idxs = np.searchsorted(self.wavelengths, self.max_primaries)
        
        chosen_primaries = np.array([matrix[:, i] for i in idxs])
        chosen_primaries = np.vstack([chosen_primaries, np.zeros(3)])
        for i, primary in enumerate(chosen_primaries):
            if i == len(chosen_primaries)-1: 
                continue
            ax.plot([0, primary[0]], [0, primary[1]], [0, primary[2]], color=np.clip(color_patches[i], 0, 1))

        # Compute the convex hull of the matrix
        hull = ConvexHull(chosen_primaries)

        # Plot the convex hull
        ax.plot_trisurf(chosen_primaries[:, 0], chosen_primaries[:, 1], chosen_primaries[:, 2], triangles=hull.simplices, color='gray', alpha=0.2)
        plt.show()