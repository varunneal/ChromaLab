from itertools import combinations
from importlib import resources
from functools import reduce
from typing import Literal

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from colour import xyY_to_XYZ, sRGB_to_XYZ
import pandas as pd

from .zonotope import getZonotopeForIntersection
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

_IN_GAMUT_TYPES = Literal["none", "clip_to_zero", "constant_ratio", "constant_luminance", "minmax_gamut"]

class ChromaticityDiagramType(Enum):
    XY=0
    ConeBasis=1
    HeringMaxBasisDisplay=2

class DisplayGamut(ABC):
    
    dim4SampleConst = 10
    dim3SampleConst = 5
        
    def __init__(self, observer, chromaticity_diagram_type=ChromaticityDiagramType.ConeBasis, transformMatrix=None, projection_idxs=None , verbose=False) -> None:
        self.verbose = verbose
        self.observer = observer
        self.chrom_diag_type = chromaticity_diagram_type
        self.wavelengths = observer.wavelengths
        self.matrix = observer.get_normalized_sensor_matrix()
        self.dimension = observer.dimension
        self.projection_idxs = list(range(1, self.dimension)) if projection_idxs is None else projection_idxs
        self.chromaticity_transform = self.get_chromaticity_conversion_function(transformMatrix, idxs=self.projection_idxs)
        self.chromaticity_mat = self.chromaticity_transform(self.matrix)
        self.chrom_intensities = None

    def get_chromaticity_conversion_function(self, transformMatrix=None, idxs=None):
        T = np.eye(self.observer.dimension)
        match self.chrom_diag_type: 
            case ChromaticityDiagramType.XY:
                if self.observer.dimension != 3:
                    raise ValueError("Chromaticity Diagram Type XY is only supported for 3 dimensions")
                raise NotImplementedError("Chromaticity Diagram Type XY is not implemented")
            
            case ChromaticityDiagramType.ConeBasis:
                return lambda pts : transformToDisplayChromaticity(pts, T, idxs=idxs)

            case ChromaticityDiagramType.HeringMaxBasisDisplay:
                if transformMatrix is not None:
                    dim = self.observer.dimension
                    def conv_chromaticity(pts):
                        vecs = (transformMatrix@pts)
                        T = getHeringMatrix(dim)
                        return transformToDisplayChromaticity(vecs, T, idxs=idxs)
                    return conv_chromaticity
                else:
                    raise ValueError("Transform Matrix is not set with ChromaticityDiagramType.HeringMaxBasis")            
            case _:
                raise ValueError("Invalid Chromaticity Diagram Type")

    def convertActivationsToIntensities(self, activations, type_: _IN_GAMUT_TYPES = "none"):
        white_point = self.observer.get_whitepoint() # d tuple
        points = np.matmul(self.M_ConeToPrimaries, activations).T
        
        if type_ == "none":
            return points
        elif type_ == "clip_to_zero":
            return np.clip(points, 0) # get rid of all negative values
        elif type_ == "constant_ratio":
            # display gamut is a convex hull of the primaries
            facets = getZonotopeForIntersection(self.primary_intensities, self.dimension)
            return getResizedGamut(points, facets) # TODO: to redo to make the white point a certain direction? 
        elif type_ == "constant_luminance":
            facets = getZonotopeForIntersection(self.primary_intensities, self.dimension)
            return getLumConstPoints(points, facets)  # TODO: to redo to make the white point a certain direction?
        elif type_ == "minmax_gamut" and type(self) == SixPrimaryDisplayGamut:
            facetsRGB = getZonotopeForIntersection(self.primary_intensities[:, :3], 3)
            facetsOCV = getZonotopeForIntersection(self.primary_intensities[:, 3:], 3)

            # individually solve for each gamut (as there is only one facet in each case, early exit is faster than searching for two faces that intersect in a group)
            # then take the min of the two solutions as you want the one that fits in both gamuts
            out1, tmaxes1 = getLumConstPoints(points, facetsRGB)
            out2, tmaxes2 = getLumConstPoints(points, facetsOCV)
            true_out = np.zeros_like(out1)
            true_out[tmaxes2 > tmaxes1] = out1[tmaxes2 > tmaxes1]
            true_out[tmaxes2 <= tmaxes1] = out2[tmaxes2 <= tmaxes1]
            return true_out
    

    def getMaxDisplayBasis(self):
        mat = np.linalg.inv(self.primary_intensities)
        # max_matrix = np.dot(self.cone_to_maxbasis, mat)
        return mat

    def setPrimariesWithSpectra(self, spectras):
        self.primary_spectras = spectras
        self.primary_intensities = np.array([self.observer.observe_normalized(s) for s in spectras])
        self.chrom_intensities = self.chromaticity_transform(self.primary_intensities.T).T
        self.primary_colors = np.clip(np.array([s.to_rgb() for s in spectras]), 0, 1)
        self.M_ConeToPrimaries = np.linalg.inv(self.primary_intensities.T)
    
    def setMonochromaticPrimaries(self, primaries):
        self.primary_spectras = [Spectra(wavelengths=self.wavelengths, data=np.zeros_like(self.wavelengths)) for _ in range(len(primaries))]
        idxs = np.searchsorted(self.wavelengths, primaries)
        for i, p in enumerate(primaries):
            self.primary_spectras[i].data[idxs[i]] = 1

        self.primary_intensities = np.array([self.matrix.T[i] for i in idxs])
        self.chrom_intensities = self.chromaticity_transform(self.primary_intensities.T).T
        self.primary_colors = np.clip(np.array([getsRGBfromWavelength(x) for x in primaries]), 0, 1)

        self.M_ConeToPrimaries = np.linalg.inv(self.primary_intensities.T)

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
            assert hasattr(self, 'primary_intensities'), "Primaries attribute is not set"
            return self.computeSimplexVolume(self.chrom_intensities)
    
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
        self.setMonochromaticPrimaries(np.array(max_primaries))
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

        self.setMonochromaticPrimaries(np.array(max_primaries))
        return max_primaries
    
    def computeProjectedConvexHull(self):
        hull = ConvexHull(self.matrix.T)
        points = self.chromaticity_transform(hull.points.T).T
        hull2 = ConvexHull(points)
        return points, hull2
    
    @abstractmethod
    def computeBarycentricCoordinates(self, coordinates, p):
       raise NotImplementedError("Abstract Method")
        
    def displayPrimariesInMaxSimplex(self):
        simplex_coords = self._genSimplex(1, self.dimension)
        coords = np.zeros((self.dimension, self.chromaticity_mat.shape[1]))
        for i in range(self.chromaticity_mat.shape[1]):
            coords[:, i] = self.computeBarycentricCoordinates(self.chrom_intensities, self.chromaticity_mat[:, i])
        
        barycentric_coords = (simplex_coords.T@coords)
        
        return simplex_coords, barycentric_coords


    @abstractmethod
    def displayPrimariesInChromDiagram(self, primaries=None): 
        raise NotImplementedError("Method not implemented")


class SixPrimaryDisplayGamut(DisplayGamut):
    
    def __init__(self, observer, factor, verbose=False) -> None:
        self.factor = factor
        self.verbose = verbose
        self.observer = observer
        self.wavelengths = observer.wavelengths
        self.matrix = observer.get_normalized_sensor_matrix()
        self.dimension = observer.dimension

    def computeBarycentricCoordinates(self):
        pass

    def displayPrimariesInChromDiagram(self):
        pass

    def mapPointsToGamut(self, points):
        return np.clip(points, 0, 1)
        facetsRGB = getZonotopeForIntersection(self.primary_intensities[:3], 3)
        facetsOCV = getZonotopeForIntersection(self.primary_intensities[3:], 3)

        # individually solve for each gamut (as there is only one facet in each case, early exit is faster than searching for two faces that intersect in a group)
        # then take the min of the two solutions as you want the one that fits in both gamuts
        out1, tmaxes1 = getLumConstPoints(points, facetsRGB)
        out2, tmaxes2 = getLumConstPoints(points, facetsOCV)
        true_out = np.zeros_like(out1)
        true_out[tmaxes2 > tmaxes1] = out1[tmaxes2 > tmaxes1]
        true_out[tmaxes2 <= tmaxes1] = out2[tmaxes2 <= tmaxes1]
        return true_out

    def setPrimariesWithSpectra(self, spectras):
        self.primary_spectras = spectras
        self.primary_intensities = np.array([self.observer.observe(s) for s in spectras]) * self.factor
        self.primary_colors = np.clip(np.array([s.to_rgb() for s in spectras]), 0, 1)


    def convert_sRGB_to_Positive_Primaries(self, sRGB):
        XYZ = sRGB_to_XYZ(sRGB)
        xyzs = np.array([s.to_xyz() for s in self.primary_spectras])
        RGB_mat = np.linalg.inv(np.array(xyzs[:3]).T)
        OCV_mat = np.linalg.inv(np.array(xyzs[3:]).T)
        return self.mapPointsToGamut(np.dot(RGB_mat, XYZ.T).T/ self.factor), self.mapPointsToGamut(np.dot(OCV_mat, XYZ.T).T / self.factor)
    
    @staticmethod
    def loadTutenLabDisplayInSixPrimaryMode(observer, factor=10000):
        with resources.path("chromalab.leds", '6primaryDLP_SPD_data_20240821.csv') as data_path:
            led_spectrums = np.array(pd.read_csv(data_path, skiprows=1)) # ordered as RGBOCV

        wavelengths = led_spectrums[:, 0]
        led_spectrums = convert_refs_to_spectras(led_spectrums[:, 1:].T, wavelengths)

        tet_disp = SixPrimaryDisplayGamut(observer, factor)
        tet_disp.setPrimariesWithSpectra(led_spectrums)
        return tet_disp

class TetraDisplayGamut(DisplayGamut):
    AXIS_LABELS = ['$S/A$', '$M/A$', '$Q/A$', '$L/A$']
    
    """
    loadTutenLabDisplay specify led_indices in array RGBOCV, and return the display object
    """
    @staticmethod
    def loadTutenLabDisplay(observer, led_indices, transform=None):
        # with resources.path("chromalab.leds", 'rgbo00.csv') as data_path:
        with resources.path("chromalab.leds", 'rgbocv-11-1.csv') as data_path:
        # with resources.path("chromalab.leds", '6primaryDLP_SPD_data_20240821.csv') as data_path:
        # with resources.path("chromalab.leds", 'r00ocv.csv') as data_path:
            led_spectrums = np.array(pd.read_csv(data_path, skiprows=1)) # ordered as RGBOCV
        # normalized_spectras = led_spectrums.copy()
        # for i in range(1, led_spectrums.shape[1]):
        #     peak = np.max(led_spectrums[:, i])
        #     normalized_spectras[:, i] = led_spectrums[:, i] * (0.010 / peak)
        wavelengths = led_spectrums[:, 0]
        led_spectrums = convert_refs_to_spectras(led_spectrums[:, np.array(led_indices) + 1].T, wavelengths)
        # lms_activations = [observer.observe(s) for s in led_spectrums]

        tet_disp = TetraDisplayGamut(observer, chromaticity_diagram_type=ChromaticityDiagramType.ConeBasis)
        tet_disp.setPrimariesWithSpectra(led_spectrums)
        return tet_disp
    
    def createDisplayFromSpectra(observer, wavelengths, spectras):
        led_spectrums = convert_refs_to_spectras(spectras, wavelengths)

        tet_disp = TetraDisplayGamut(observer, chromaticity_diagram_type=ChromaticityDiagramType.ConeBasis)
        tet_disp.setPrimariesWithSpectra(led_spectrums)
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

    def displayPrimariesInMaxSimplex(self):
        simplex_coords, barycentric_coords = super().displayPrimariesInMaxSimplex()

        hull = ConvexHull(simplex_coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(simplex_coords[:, 0], simplex_coords[:, 1], simplex_coords[:, 2], triangles=hull.simplices, color='gray', alpha=0.2)
        ax.scatter(simplex_coords[:, 0], simplex_coords[:, 1], simplex_coords[:, 2], c=self.primary_colors, s=100)
        ax.plot(barycentric_coords[0], barycentric_coords[1], barycentric_coords[2], c='b', alpha=0.5)
        plt.gca().set_aspect('equal')
        plt.show()
        
        

    def displayPrimariesInChromDiagram(self, primaries=None, title=None):
        fig = plt.figure(figsize=plt.figaspect(0.33))

        ax = fig.add_subplot(1, 3, 1)
        ax.set_title("Spectral Locus in Chromaticity")
        ax.set_xlim(self.wavelengths[0], self.wavelengths[-1])
        ax.set_xlabel('Wavelengths')
        ax.set_ylabel('Sensitivity')
        colors = ['b', 'g', 'r'] if self.dimension == 3 else ['b', 'g', 'y', 'r']
        # for i, x in enumerate(max_primaries):
        #     ax.axvline(x, color=self.primary_colors[i])
        
        # for j in range(self.observer.dimension):
        #     ax.plot(self.wavelengths, self.observer.get_sensor_matrix()[j], c=colors[j], alpha=0.5)

        for j in range(self.chromaticity_mat.shape[0]):
            ax.plot(self.wavelengths, self.chromaticity_mat[j], c=self.primary_colors[j], alpha=0.5)

        matrix = self.chromaticity_mat if primaries is None else primaries
        ax1 = fig.add_subplot(1, 3, 2, projection='3d')
        if title is not None: 
            ax1.set_title(title)
        ax1.plot(matrix[0], matrix[1], matrix[2])
        ax1.set_xlabel(TetraDisplayGamut.AXIS_LABELS[self.projection_idxs[0]])
        ax1.set_ylabel(TetraDisplayGamut.AXIS_LABELS[self.projection_idxs[1]])
        ax1.set_zlabel(TetraDisplayGamut.AXIS_LABELS[self.projection_idxs[2]])

        ax1.scatter(self.chrom_intensities[:, 0], self.chrom_intensities[:, 1], self.chrom_intensities[:, 2], c=self.primary_colors, s=100)

        # Compute & plot the convex hull of the matrix
        hull1 = ConvexHull(self.chrom_intensities)
        ax1.plot_trisurf(self.chrom_intensities[:, 0], self.chrom_intensities[:, 1], self.chrom_intensities[:, 2], triangles=hull1.simplices, color='gray', alpha=0.2)

        # Ideal Hull
        ax2 = fig.add_subplot(1, 3, 3, projection='3d')
        ax2.set_title(f"Spectral Display")
        ax2.plot(matrix[0], matrix[1], matrix[2])
        ax1.set_xlabel(TetraDisplayGamut.AXIS_LABELS[self.projection_idxs[0]])
        ax1.set_ylabel(TetraDisplayGamut.AXIS_LABELS[self.projection_idxs[1]])
        ax1.set_zlabel(TetraDisplayGamut.AXIS_LABELS[self.projection_idxs[2]])

        _, hull2 = self.computeProjectedConvexHull()
        ax2.plot_trisurf(hull2.points[:, 0], hull2.points[:, 1], hull2.points[:, 2], triangles=hull2.simplices, color='gray', alpha=0.2)
        ax1.shareview(ax2)
        print(f"Volume Ratio Between n primaries / ideal = {hull1.volume/hull2.volume}")
        plt.tight_layout()
        plt.show()


class TriDisplayGamut(DisplayGamut):
    AXIS_LABELS = ['$S/A$', '$M/A$', '$L/A$']

    @staticmethod
    def loadTutenLabDisplay(observer, led_indices, transform=None):
        with resources.path("chromalab.leds", '6primaryDLP_SPD_data_20240821.csv') as data_path:
        # with resources.path("chromalab.leds", 'r00ocv.csv') as data_path:
            led_spectrums = np.array(pd.read_csv(data_path, skiprows=1)) # ordered as RGBOCV

        wavelengths = led_spectrums[:, 0]
        led_spectrums = convert_refs_to_spectras(led_spectrums[:, np.array(led_indices) + 1].T, wavelengths)
        # lms_activations = [observer.observe(s) for s in led_spectrums]

        tri_disp = TriDisplayGamut(observer, chromaticity_diagram_type=ChromaticityDiagramType.ConeBasis)
        tri_disp.setPrimariesWithSpectra(led_spectrums)
        return tri_disp

    def computeBarycentricCoordinates(self, coordinates, p):
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

    def displayPrimariesInMaxSimplex(self):
        simplex_coords, barycentric_coords = super().displayPrimariesInMaxSimplex()
        hull = ConvexHull(simplex_coords)

        fig, ax = plt.subplots()
        self.__plot_2d_hull(simplex_coords, hull, ax, 'k', 0.2)
        ax.scatter(simplex_coords[:, 0], simplex_coords[:, 1], c=self.primary_colors, s=100)
        ax.plot(barycentric_coords[0], barycentric_coords[1], c='b', alpha=0.5)
        ax.set_aspect('equal')
        plt.show()

    def displayPrimariesInChromDiagram(self, primaries=None, title=None):
        fig = plt.figure(figsize=plt.figaspect(0.33))

        max_primaries = self.chrom_intensities if primaries is None else primaries
        ax = fig.add_subplot(1, 3, 1)
        ax.set_title("Spectral Locus in Chromaticity")
        ax.set_xlim(self.wavelengths[0], self.wavelengths[-1])
        ax.set_xlabel('Wavelengths')
        ax.set_ylabel('Sensitivity')

        # for i, x in enumerate(max_primaries):
        #     ax.axvline(x, color=self.primary_colors[i])

        for j in range(self.chromaticity_mat.shape[0]):
            ax.plot(self.wavelengths, self.chromaticity_mat[j], c=self.primary_colors[j], alpha=0.5)
        
        # colors = ['b', 'g', 'r'] if self.dimension == 3 else ['b', 'g', 'y', 'r']
        # for j in range(self.observer.dimension):
        #     ax.plot(self.wavelengths, self.observer.get_sensor_matrix()[j], c=colors[j], alpha=0.5)

        matrix = self.chromaticity_mat if primaries is None else primaries
        ax = fig.add_subplot(1, 3, 2)
        if title is not None:
            ax.set_title(title)
        ax.plot(matrix[0], matrix[1])
        
        hull1 = ConvexHull(self.chrom_intensities)
        # convex_hull_plot_2d(hull, ax)
        self.__plot_2d_hull(self.chrom_intensities, hull1, ax, 'k', 0.2)
        ax.scatter(self.chrom_intensities[:, 0], self.chrom_intensities[:, 1], c=self.primary_colors, s=100)
        ax.set_xlabel(TriDisplayGamut.AXIS_LABELS[self.projection_idxs[0]])
        ax.set_ylabel(TriDisplayGamut.AXIS_LABELS[self.projection_idxs[1]])

        # Ideal Hull 
        ax = fig.add_subplot(1, 3, 3)
        ax.set_title(f"Maximum Possible Display Gamut")
        ax.plot(matrix[0], matrix[1])
        points, hull2 = self.computeProjectedConvexHull()
        print(points.shape)
        self.__plot_2d_hull(points, hull2, ax, 'k', 0.2)
        ax.scatter(points[:, 0], points[:, 1], c='orange', s=100)
        ax.set_xlabel(TriDisplayGamut.AXIS_LABELS[self.projection_idxs[0]])
        ax.set_ylabel(TriDisplayGamut.AXIS_LABELS[self.projection_idxs[1]])
        # self._plot_sRGB_Gamut(ax)

        print(f"Volume Ratio Between n primaries / ideal = {hull1.volume/hull2.volume}")
        plt.tight_layout()
        plt.show()

def computeParalleltopeCoords(ray_dir, points, ray_origin=None):
    """
    Compute the alpha, beta, and gamma that correspond to a point relative to a parallelotope 
    defined by the points P0 (origin of parallelotope), P1 (dir 1), P2, and P3.

    ray_dir is a d-dim vector. 
    ray_origin is a d-dim point. 
    points is a d x d matrix of points, where each row is a point in d-dim space 
    """

    if ray_origin is None:
        ray_origin = np.zeros_like(ray_dir)
    
    dim = len(ray_dir)
    # Compute the matrix M that has the points as columns
    M = np.zeros((dim, dim))
    M[:, 0] = -ray_dir
    for i in range(1, dim):
        M[:, i] = points[i] - points[0]

    b = ray_origin - points[0]
    if np.linalg.matrix_rank(M) < dim:
        return None
    vecs = np.linalg.solve(M, b)
    return vecs

def approx_lte(x, y):
    return np.logical_or(x <= y, np.isclose(x, y))
def approx_gte(x, y):
    return np.logical_or(x >= y, np.isclose(x, y))

def getResizedGamut(candidate_points, paralleletope_facets, ray_origin=None):
    """
    Return the t parameter of a ray direction such that it lies inside of the parallelotope defined by P0, P1, P2, and P3. 
    """

    if ray_origin is None:
        ray_origin = np.zeros_like(candidate_points[0])

    current_multiple = 1
    for candidate_point in candidate_points:
        t = np.linalg.norm(candidate_point)
        ray_dir = candidate_point/t
        for paralleletope_facet in paralleletope_facets:
            vecs = computeParalleltopeCoords(ray_dir, paralleletope_facet, ray_origin)
            # the intersection being less than t means that the point is outside the gamut
            if vecs is not None and np.all(vecs[1:] >= 0) and np.all(vecs[1:] <= 1) and vecs[0] < t and vecs[0] > 0:
                current_multiple = min(current_multiple, vecs[0]/t)
    return current_multiple
    
def getLumConstPoints(candidate_points, paralleletope_facets):
    """
    Remap points to fit in gamut by projecting onto the luminance axis and then choosing the most saturated
    candidate points --  Nxd array where N is the number of points and d is dimension
    paralleletope_facets -- list of [o, v, w] vectors that define a sub-paralleletope of dimension d-1
    """
    lum_dir = np.ones(candidate_points.shape[1])

    # check if the max point is larger than white, if so, we need to scale down the gamut
    norms_on_lum = np.max(np.dot(candidate_points, lum_dir.T)/np.linalg.norm(lum_dir)**2)
    if norms_on_lum > 1:
        multiple = getResizedGamut(candidate_points, paralleletope_facets)
        candidate_points = candidate_points * multiple

    res_pts = []
    tmaxes = []
    for i, candidate_point in enumerate(candidate_points):
        projection = np.dot(candidate_point, lum_dir.T) / np.linalg.norm(lum_dir)**2
        ray_origin = projection * lum_dir
        ray_dir = candidate_point - ray_origin
        for paralleletope_facet in paralleletope_facets:
            vecs = computeParalleltopeCoords(ray_dir, paralleletope_facet, ray_origin)
            # the intersection being less than t means that the point is outside the gamut
            if vecs is not None and np.all(approx_gte(vecs[1:],0)) and np.all(approx_lte(vecs[1:],1)) and vecs[0] >= 0:
                # print(f"direction intersects with gamut --  ray_origin {ray_origin} ray_dir {ray_dir} resulting intersection{vecs}\n")
                res_pts +=[vecs[0] * ray_dir + ray_origin]
                tmaxes += [vecs[0]]
                break
    if len(res_pts) != len(candidate_points):
        raise Exception(f"Some points were not projected onto the gamut, issue with algorithm? {res_pts}")
    
    return np.array(res_pts), np.array(tmaxes)