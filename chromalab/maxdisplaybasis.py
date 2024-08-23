from itertools import combinations
from importlib import resources
from functools import reduce

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from colour import xyY_to_XYZ
import pandas as pd

from .spectra import Spectra, convert_refs_to_spectras
from .observer import getsRGBfromWavelength, transformToDisplayChromaticity, transformToChromaticity, getHeringMatrix
from scipy.spatial import ConvexHull, convex_hull_plot_2d


def plotAxisAlignedProjections4D(points):

    fig = plt.figure(figsize=(8, 5))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])

    ax = fig.add_subplot(gs[0:2, 0])
    ax.plot(points)
    ax.set_xlabel('Wavelengths')

    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], '-o', alpha=0.6, c=np.arange(len(points)), cmap='coolwarm')
    ax2.set_xlabel('$S/A$')
    ax2.set_ylabel('$M/A$')
    ax2.set_zlabel('$Q/A$')

    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax3.scatter(points[:, 0], points[:, 1], points[:, 3], '-o', alpha=0.6,  c=np.arange(len(points)), cmap='coolwarm')
    ax3.set_xlabel('$S/A$')
    ax3.set_ylabel('$M/A$')
    ax3.set_zlabel('$L/A$')

    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    ax4.scatter(points[:, 0], points[:, 2], points[:, 3], '-o', alpha=0.6,  c=np.arange(len(points)), cmap='coolwarm')
    ax4.set_xlabel('$S/A$')
    ax4.set_ylabel('$Q/A$')
    ax4.set_zlabel('$L/A$')

    ax5 = fig.add_subplot(gs[1, 2], projection='3d')
    ax5.scatter(points[:, 1], points[:, 2], points[:, 3], '-o', alpha=0.6,  c=np.arange(len(points)), cmap='coolwarm')
    ax5.set_xlabel('$M/A$')
    ax5.set_ylabel('$Q/A$')
    ax5.set_zlabel('$L/A$')

    # plt.tight_layout()
    plt.show()

def plotAxisAlignedProjections(points, axis_labels=['$B$', '$G$', '$R$']):

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[2, 1], width_ratios=[1, 1, 1])

    # Axis aligned projections
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(points, '-o', markersize=5, alpha=0.6)

    ax1 = fig.add_subplot(gs[0, 1:3], projection='3d')
    ax1.plot(points[:, 0], points[:, 1], points[:, 2], '-o', markersize=5, alpha=0.6)
    ax1.set_xlabel(axis_labels[0])
    ax1.set_ylabel(axis_labels[1])
    ax1.set_zlabel(axis_labels[2])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(points[:, 0], points[:, 1], '-o', markersize=5, alpha=0.6)
    ax2.set_xlabel(axis_labels[0])
    ax2.set_ylabel(axis_labels[1])

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(points[:, 1], points[:, 2], '-o', markersize=5, alpha=0.6)
    ax3.set_xlabel(axis_labels[1])
    ax3.set_ylabel(axis_labels[2])

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(points[:, 0], points[:, 2], '-o', markersize=5, alpha=0.6)
    ax4.set_xlabel(axis_labels[0])
    ax4.set_ylabel(axis_labels[2])

    plt.tight_layout()
    plt.show()


class ChromaticityDiagramType(Enum):
    XY=0
    LaserPoint=1
    HeringMaxBasisDisplay=2

class DisplayGamut(ABC):
    
    dim4SampleConst = 10
    dim3SampleConst = 5
        
    def __init__(self, observer, chromaticity_diagram_type=ChromaticityDiagramType.LaserPoint, transformMatrix=None, verbose=False) -> None:
        self.verbose = verbose
        self.observer = observer
        self.chrom_diag_type = chromaticity_diagram_type
        self.wavelengths = observer.wavelengths
        self.matrix = observer.get_normalized_sensor_matrix()
        self.dimension = observer.dimension
        self.chromaticity_transform = self.get_chromaticity_conversion_function(transformMatrix)
        self.chromaticity_mat = self.chromaticity_transform(self.matrix)
        self.wavelength_primaries = None
        self.primary_intensities = None

    def get_chromaticity_conversion_function(self, transformMatrix=None):
        T = np.eye(self.observer.dimension)
        match self.chrom_diag_type: 
            case ChromaticityDiagramType.XY:
                if self.observer.dimension != 3:
                    raise ValueError("Chromaticity Diagram Type XY is only supported for 3 dimensions")
                raise NotImplementedError("Chromaticity Diagram Type XY is not implemented")
            
            case ChromaticityDiagramType.LaserPoint:
                return lambda pts : transformToDisplayChromaticity(pts, T)

            case ChromaticityDiagramType.HeringMaxBasisDisplay:
                if transformMatrix is not None:
                    dim = self.observer.dimension
                    def conv_chromaticity(pts):
                        vecs = (transformMatrix@pts)
                        T = getHeringMatrix(dim)
                        return transformToDisplayChromaticity(vecs, T)
                    return conv_chromaticity
                else:
                    raise ValueError("Transform Matrix is not set with ChromaticityDiagramType.HeringMaxBasis")            
            case _:
                raise ValueError("Invalid Chromaticity Diagram Type")

    def convertActivationsToIntensities(self, activations):
        return np.matmul(np.linalg.inv(self.primary_intensities), activations)
    
    def setMonochromaticPrimaries(self, primaries):

    
    def setPrimaries(self, primaries):
        # assert len(primaries) == self.dimension
        assert primaries.shape[1] == (self.dimension-1)
        self.primaries = primaries

    def computeSimplexVolume(self, wavelengths):
        idxs = np.searchsorted(self.wavelengths, wavelengths) # pick index wavelengths from list of wavelengths
        mat = np.ones((self.dimension, self.dimension))
        submat = self.chromaticity_mat[:, idxs] # list of basis vectors per index in cone space
        mat[:, 1:] = submat.T
        fact = reduce(lambda x, y: x*y, range(1, self.dimension + 1))
        vol = np.abs(np.linalg.det(mat))/fact
        return vol

    def computeVolumeOfPrimaries(self, primaries=None):
        if primaries is not None:
            return self.computeSimplexVolume(primaries)
        else:
            assert hasattr(self, 'primaries'), "Primaries attribute is not set"
            return self.computeSimplexVolume(self.primaries)
    
    def computeMaxPrimariesInFull(self):

        def computeVolume(wavelengths): # only works in the self.dimension case cuz it's a parallelotope
            # set of n wavelengths, and we want to pick the monochromatic wavelength in each direction
            idxs = np.searchsorted(self.wavelengths, wavelengths) # pick index wavelengths from list of wavelengths
            submat = self.matrix[:, idxs] # list of basis vectors per index in cone space
            vol = np.linalg.det(submat) # volume of resulting parallelepiped -> according to chatGPT, it is divided by a fixed 1/n! factor based on dimensionality, so doesn't matter if we divide or not. 
            return vol

        data = list(combinations(self.wavelengths, self.dimension))
        result = list(map(computeVolume, data))
        idx = np.argmax(result)
        max_primaries = list(data)[idx]
        print(f"Max Primaries -- {max_primaries}")
        print(f"Max Volume = {result[idx]}")
        self.primaries = max_primaries
        return max_primaries

    @staticmethod
    def _genSimplex(side_length, dimension):
        # Calculate the height of the tetrahedron
        height = (3 ** 0.5) * side_length / 2

        # Calculate the coordinates of the vertices
        vertex1 = [-side_length/2, 0, 0]
        vertex2 = [side_length/2, 0, 0]
        vertex3 = [0, height, 0]
        vertex4 = [0, height/3, (2/3)**0.5 * side_length]

        arr = np.array([vertex1, vertex2, vertex3, vertex4])

        if dimension == 3:
            return arr[:3, :2]
        elif dimension == 4:
            return arr
        else:
            raise ValueError("Only dimensions 3 & 4 are supported")

    def computeMaxPrimariesInChrom(self, wavelengths=None):
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        data = list(combinations(wavelengths, self.dimension))
        result = list(map(self.computeSimplexVolume, data))
        idx = np.argmax(result)
        max_primaries = list(data)[idx]
        print(f"Max Primaries -- {max_primaries}")
        print(f"Max Volume = {result[idx]}")

        self.primaries = max_primaries
        return max_primaries
    
    def computeProjectedConvexHull(self):
        hull = ConvexHull(self.matrix.T)
        points = transformToDisplayChromaticity(hull.points.T, self.T).T
        hull2 = ConvexHull(points)
        return points, hull2
    
    @abstractmethod
    def computeBarycentricCoordinates(self, coordinates, p):
       raise NotImplementedError("Abstract Method")
        
    def displayPrimariesInMaxSimplex(self, primaries=None):
        # TODO: fix primaries attribute to separate between the actual chromaticity value, and the wavelength (cuz wavelength only works for monochromatic lights)
        if primaries is None and not hasattr(self, 'primaries'):
            primaries = self.computeMaxPrimariesInChrom()
        else:
            primaries = self.primaries if primaries is None else primaries
        
        simplex_coords = self._genSimplex(1, self.dimension)
        idxs = np.searchsorted(self.wavelengths, primaries)
        chosen_primaries = np.array([self.chromaticity_mat[:, i] for i in idxs])

        coords = np.zeros((self.dimension, self.chromaticity_mat.shape[1]))
        for i in range(self.chromaticity_mat.shape[1]):
            coords[:, i] = self.computeBarycentricCoordinates(chosen_primaries, self.chromaticity_mat[:, i])
        
        barycentric_coords = (simplex_coords.T@coords)
        
        return simplex_coords, barycentric_coords


    @abstractmethod
    def displayPrimariesInChromDiagram(self, primaries=None): 
        raise NotImplementedError("Method not implemented")

class TetraDisplayGamut(DisplayGamut):
    
    """
    loadTutenLabDisplay specify led_indices in array RGBOCV, and return the display object
    """
    @staticmethod
    def loadTutenLabDisplay(observer, led_indices, transform=None):
        with resources.path("chromalab.leds", '6primaryDLP_SPD_data_20240821.csv') as data_path:
            led_spectrums = np.array(pd.read_csv(data_path, skiprows=1)) # ordered as RGBOCV

        wavelengths = led_spectrums[:, 0]
        led_spectrums = convert_refs_to_spectras(led_spectrums[:, np.array(led_indices) + 1].T, wavelengths)
        lms_activations = [observer.observe(s) for s in led_spectrums]

        tet_disp = TetraDisplayGamut(observer, chromaticity_diagram_type=ChromaticityDiagramType.LaserPoint)
        
        pts_in_chrom = tet_disp.chromaticity_transform(lms_activations)
        tet_disp.setPrimaries(pts_in_chrom.T)

        return tet_disp

    def computeBarycentricCoordinates(self, coordinates, p):
        a = coordinates[0]
        b = coordinates[1]
        c = coordinates[2]
        d = coordinates[3]

        vap = p - a
        vbp = p - b

        vab = b - a
        vac = c - a
        vad = d - a

        vbc = c - b
        vbd = d - b

        va6 = np.dot(np.cross(vbp, vbd), vbc)
        vb6 = np.dot(np.cross(vap, vac), vad)
        vc6 = np.dot(np.cross(vap, vad), vab)
        vd6 = np.dot(np.cross(vap, vab), vac)
        v6 = 1 / np.dot(np.cross(vab, vac), vad)
        return np.array([va6*v6, vb6*v6, vc6*v6, vd6*v6])

    def displayPrimariesInMaxSimplex(self, primaries=None):
        simplex_coords, barycentric_coords = super().displayPrimariesInMaxSimplex(primaries)

        color_patches = np.clip([getsRGBfromWavelength(x) for x in self.primaries], 0, 1)
        hull = ConvexHull(simplex_coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(simplex_coords[:, 0], simplex_coords[:, 1], simplex_coords[:, 2], triangles=hull.simplices, color='gray', alpha=0.2)
        ax.scatter(simplex_coords[:, 0], simplex_coords[:, 1], simplex_coords[:, 2], c=color_patches, s=100)
        ax.plot(barycentric_coords[0], barycentric_coords[1], barycentric_coords[2], c='b', alpha=0.5)
        ax.set_box_aspect([1, 1, 1])
        plt.show()
        
        

    def displayPrimariesInChromDiagram(self, primaries=None, title=None):
        fig = plt.figure(figsize=plt.figaspect(0.33))

        max_primaries = self.primaries if primaries is None else primaries
        color_patches = np.clip([getsRGBfromWavelength(x) for x in max_primaries], 0, 1)

        ax = fig.add_subplot(1, 3, 1)
        ax.set_title("Monochromatic Wavelengths")
        ax.set_xlim(self.wavelengths[0], self.wavelengths[-1])
        ax.set_xlabel('Wavelengths')
        ax.set_ylabel('Sensitivity')
        colors = ['b', 'g', 'r'] if self.dimension == 3 else ['b', 'g', 'y', 'r']
        for i, x in enumerate(max_primaries):
            ax.axvline(x, color=color_patches[i])
        
        for j in range(self.observer.dimension):
            ax.plot(self.wavelengths, self.observer.get_sensor_matrix()[j], c=colors[j], alpha=0.5)

        matrix = self.chromaticity_mat if primaries is None else primaries
        ax1 = fig.add_subplot(1, 3, 2, projection='3d')
        if title is not None: 
            ax1.set_title(title)
        ax1.plot(matrix[0], matrix[1], matrix[2])
        ax1.set_xlabel('$M/A$')
        ax1.set_ylabel('$Q/A$')
        ax1.set_zlabel('$L/A$')

        idxs = np.searchsorted(self.wavelengths, max_primaries)
        
        chosen_primaries = np.array([matrix[:, i] for i in idxs])
        ax1.scatter(chosen_primaries[:, 0], chosen_primaries[:, 1], chosen_primaries[:, 2], c=color_patches, s=100)

        # Compute & plot the convex hull of the matrix
        hull1 = ConvexHull(chosen_primaries)
        ax1.plot_trisurf(chosen_primaries[:, 0], chosen_primaries[:, 1], chosen_primaries[:, 2], triangles=hull1.simplices, color='gray', alpha=0.2)

        # Ideal Hull
        ax2 = fig.add_subplot(1, 3, 3, projection='3d')
        ax2.set_title(f"Spectral Display")
        ax2.plot(matrix[0], matrix[1], matrix[2])
        ax2.set_xlabel('$M/A$')
        ax2.set_ylabel('$Q/A$')
        ax2.set_zlabel('$L/A$')

        _, hull2 = self.computeProjectedConvexHull()
        ax2.plot_trisurf(hull2.points[:, 0], hull2.points[:, 1], hull2.points[:, 2], triangles=hull2.simplices, color='gray', alpha=0.2)
        ax1.shareview(ax2)
        print(f"Volume Ratio Between n primaries / ideal = {hull1.volume/hull2.volume}")
        plt.tight_layout()
        plt.show()


class TriDisplayGamut(DisplayGamut):

    def computeBarycentricCoordinates(coordinates, p):
        v0 = coordinates[1] - coordinates[0]
        v1 = coordinates[2] - coordinates[0]
        v2 = p - coordinates[0]

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01

        barycentric_coords = np.zeros(3)
        barycentric_coords[1] = (d11 * d20 - d01 * d21) / denom
        barycentric_coords[2] = (d00 * d21 - d01 * d20) / denom
        barycentric_coords[0] = 1 - barycentric_coords[1] - barycentric_coords[2]

        return barycentric_coords

    @staticmethod
    def __plot_2d_hull(points, hull, ax, color, opacity):
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

    # TODO: need to change the XYZ to SML matrix to match the exact cone fundamentals I use -- this probably extends to 830 which is why it's fucking up my range
    def _plot_sRGB_Gamut(self, ax):
        M_XYZ_to_SML = np.flip(np.array(
            [
                [0.4002, 0.7075, -0.0807],
                [-0.2280, 1.1500, 0.0612],
                [0.0000, 0.0000, 0.9184],
            ]), axis=0)
        xyY = np.array([[0.6400, 0.3000, 0.1500], [0.3300, 0.6000, 0.0600], [0.2126, 0.7152, 0.0722]]).T
        pts = transformToDisplayChromaticity((M_XYZ_to_SML@(xyY_to_XYZ(xyY))).T, np.eye(3)).T
        print(xyY)
        print(xyY_to_XYZ(xyY))
        print((M_XYZ_to_SML@xyY_to_XYZ(xyY)))
        print(pts)
        hull1 = ConvexHull(pts)
        ax.scatter(pts[:, 0], pts[:, 1], c='blue', s=100)
        self.__plot_2d_hull(pts, hull1, ax, 'k', 0.2)

    def displayPrimariesInMaxSimplex(self, primaries=None):
        simplex_coords, barycentric_coords = super().displayPrimariesInMaxSimplex(primaries)

        color_patches = np.clip([getsRGBfromWavelength(x) for x in self.primaries], 0, 1)
        hull = ConvexHull(simplex_coords)

        fig, ax = plt.subplots()
        self.__plot_2d_hull(simplex_coords, hull, ax, 'k', 0.2)
        ax.scatter(simplex_coords[:, 0], simplex_coords[:, 1], c=color_patches, s=100)
        ax.plot(barycentric_coords[0], barycentric_coords[1], c='b', alpha=0.5)
        ax.set_aspect('equal')
        plt.show()

    def displayPrimariesInChromDiagram(self, primaries=None, title=None):
        fig = plt.figure(figsize=plt.figaspect(0.33))

        max_primaries = self.primaries if primaries is None else primaries
        color_patches = np.clip([getsRGBfromWavelength(x) for x in max_primaries], 0, 1)

        ax = fig.add_subplot(1, 3, 1)
        ax.set_title("Monochromatic Wavelengths")
        ax.set_xlim(self.wavelengths[0], self.wavelengths[-1])
        ax.set_xlabel('Wavelengths')
        ax.set_ylabel('Sensitivity')

        for i, x in enumerate(max_primaries):
            ax.axvline(x, color=np.clip(color_patches[i], 0, 1))
        
        colors = ['b', 'g', 'r'] if self.dimension == 3 else ['b', 'g', 'y', 'r']
        for j in range(self.observer.dimension):
            ax.plot(self.wavelengths, self.observer.get_sensor_matrix()[j], c=colors[j], alpha=0.5)

        matrix = self.chromaticity_mat if primaries is None else primaries
        ax = fig.add_subplot(1, 3, 2)
        if title is not None:
            ax.set_title(title)
        ax.plot(matrix[0], matrix[1])
        idxs = np.searchsorted(self.wavelengths, max_primaries)
        points = self.chromaticity_mat[:, idxs].T
        hull1 = ConvexHull(points)
        # convex_hull_plot_2d(hull, ax)
        self.__plot_2d_hull(points, hull1, ax, 'k', 0.2)
        ax.scatter(points[:, 0], points[:, 1], c=color_patches, s=100)
        ax.set_xlabel('$M/A$')
        ax.set_ylabel('$L/A$')

        # Ideal Hull 
        ax = fig.add_subplot(1, 3, 3)
        ax.set_title(f"Maximum Possible Display Gamut")
        ax.plot(matrix[0], matrix[1])
        points, hull2 = self.computeProjectedConvexHull()
        print(points.shape)
        self.__plot_2d_hull(points, hull2, ax, 'k', 0.2)
        ax.scatter(points[:, 0], points[:, 1], c='orange', s=100)
        ax.set_xlabel('$M/A$')
        ax.set_ylabel('$L/A$')
        # self._plot_sRGB_Gamut(ax)

        print(f"Volume Ratio Between n primaries / ideal = {hull1.volume/hull2.volume}")
        plt.tight_layout()
        plt.show()