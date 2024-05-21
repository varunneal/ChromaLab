from scipy.spatial import KDTree
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from .observer import Observer, transformToChromaticity
from .spectra import Spectra
from .maxbasis import MaxBasis

def sampleCubeMapFaces(list_of_faces, samples_per_line=5):
    step = 1/samples_per_line
    xi = np.arange(0, 1, step) + (1/2*step)
    output = []
    for face in list_of_faces: # bilinear interpolation of a square face
        one_face = []
        for i in xi:
            for j in xi:
                e =  face[0] + i * (face[1] - face[0]);
                f = face[2] + i * (face[3] - face[2]);
                p = e + j*(f - e);
                one_face +=[p]
        output += [one_face]
    return np.array(output)


class CubeMap:
    """
    CubeMap class to generate a cube map from a point cloud and a set of associated rgbs or refs/wavelengths, in the Maxbasis space.
    """
    def __init__(self, point_cloud, maxBasis: MaxBasis, rgbs=None, refs=None, ref_wavelengths=None, verbose: bool=False) -> None:
        if maxBasis.dimension != 4:
            raise ValueError("Observer must be 4-dimensional")
        self.maxbasis = maxBasis
        if rgbs is not None:
            self.rgbs = rgbs
        elif refs is not None and ref_wavelengths is not None:
            self.refs = refs
            self.ref_wavelengths = ref_wavelengths
        else: 
            raise ValueError("Either rgbs or refs and ref_wavelengths must be provided")
        self.point_cloud = point_cloud
        self.verbose = verbose

    
    def __get_corners(self):
        # center, then the 4 corners of the face
        list_corners = [[6, 11, 3, 1, 13], [5, 12, 2, 1, 11], [8, 2, 14, 11, 3], [10, 14, 4, 3, 13], [7, 4, 12, 13, 1], [9, 12, 4, 2, 14]]
        refs, maxbasispoints, rgbs, lines = self.maxbasis.getDiscreteRepresentation()
        points = transformToChromaticity(maxbasispoints)
        pts_corners = points[list_corners]
        rgb_corners = rgbs[list_corners]
        return pts_corners, rgb_corners

    def __sample_cube_map(self, satval=1.0, side_len=9):
        d = 6
        cube_pts, _ = self.__get_corners()
        all_directions = sampleCubeMapFaces(cube_pts[:, 1:], side_len) # get chromaticity coord of corners
        for j in range(d):
            all_directions[j] = all_directions[j] / np.repeat(np.expand_dims(np.linalg.norm(all_directions[j], axis=1), axis=1), repeats=3, axis=1) * satval # normalize them
        return all_directions
    
    def __find_closest_points(self, candidate_points, lumval, samples_per_point=25, radius_limit=0.5, lum_thresh=0.8, out_of_range_color=[0.75, 0.75, 0.75]):
        # for each point, find top samples_per_point closest points, and figure out which one is closest to the luminance value
        kdtree = KDTree(self.point_cloud)
        dd, ii = kdtree.query(candidate_points, samples_per_point, distance_upper_bound=radius_limit)
        dd, ii = dd.reshape(-1, samples_per_point), ii.reshape(-1, samples_per_point)
        final_idxs = []
        updated_dd = []
        final_rgbs = []
        for i in tqdm(range(len(ii)), disable=not self.verbose): # set of reflectances for each point, pick closeset
            idxs = [idx for idx in ii[i] if idx != self.point_cloud.shape[0]]
            dists = [d for d in dd[i] if d != np.inf]
            
            # if no points are found, set to dummy value
            if len(idxs) == 0: 
                final_idxs += [0]
                updated_dd += [np.inf]
                final_rgbs += [out_of_range_color]
                print(f"Point {i} has no close points")
                continue

            if hasattr(self, "rgbs"):
                rgb = self.rgbs[idxs]
            else:
                rgb = np.array([Spectra(np.concatenate([self.ref_wavelengths[:, np.newaxis], self.refs[idx][:, np.newaxis]], axis=1)).to_rgb() for idx in idxs])

            min_idx = min(range(len(rgb)), key=lambda j: abs(np.sum(rgb[j])-lumval))
            if abs(np.sum(rgb[min_idx])-lumval) > lum_thresh:
                final_idxs += [0] #dummy val
                updated_dd += [np.inf]
                final_rgbs += [out_of_range_color]
                print(f"Point {i} has no close points in luminance range")
                continue
            
            final_idxs += [idxs[min_idx]]
            updated_dd += [dists[min_idx]]
            final_rgbs += [rgb[min_idx]]
        return np.array(final_idxs), np.array(updated_dd), np.array(final_rgbs)

    
    def get_cube_map(self, lumval, satval, side_len, lum_thresh=0.8, sat_thresh=0.3):
        candidate_points = self.__sample_cube_map(satval, side_len).reshape(-1, 3)
        ii, dd, final_rgbs =  self.__find_closest_points(candidate_points, lumval, samples_per_point=10, radius_limit=sat_thresh, lum_thresh=lum_thresh)
        return ii.reshape(6, side_len * side_len), dd.reshape(6, side_len * side_len), final_rgbs.reshape(6, side_len * side_len, 3)

    def display_cube_map(self, lumval, satval, side_len, lum_thresh=0.8, sat_thresh=0.3):
        idxs, distances, rgb = self.get_cube_map(lumval, satval, side_len, lum_thresh, sat_thresh)
        fig, ax = plt.subplots(figsize=(6,4))
        plt.axis('off')
        plt.gca().set_aspect('equal')
        plt.tight_layout()

        cube_left_corner_pts = np.array([[1, 2], [0, 1], [1, 1], [2, 1], [3, 1], [1, 0]])

        step = 1/side_len
        # sample in the center of the square of each sub square
        square = np.array([[i, j] for i in (np.arange(0, 1, step) + (1/2*step)) for j in np.arange(0, 1, step) + (1/2*step) ]).reshape(side_len, side_len, 2)
        cubemap = np.array([ x + square for x in cube_left_corner_pts])

        cm = cubemap.reshape(-1, 2)
        ax.scatter(cm[:, 0], cm[:, 1], c=rgb.reshape(-1, 3), s=70)
        plt.show()