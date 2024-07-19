import os
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import heapq
import h5py

from sklearn import decomposition
from scipy.ndimage import shift
from tqdm import tqdm

from chromalab.spectra import Spectra, Illuminant, convert_refs_to_rgbs
from chromalab.observer import transformToChromaticity

from abc import ABC, abstractmethod

class DataLoader(ABC):
    def __init__(self, root_path, observer):
        self.observer = observer
        self.root = root_path
    
    @abstractmethod
    def getConeResponses(self, isWhitened=True):
        pass
    
    @abstractmethod
    def getSpatialConeResponses(self, isWhitened=True):
        pass
    
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        return pickle.load(open(filename, 'rb'))


# TODO: Add Bird Spectra Data Loader
class PlumageDataLoader(DataLoader):
    def __init__(self, root_path, observer):
        self.observer = observer
        self.root = root_path
        self.data_range = [1, 900]
        data = np.load(self.root, allow_pickle=True)
        

        self.wavelengths = data['wavelengths']
        self.spectral_names = data['spectra_names']
        self.per_bird_data = data['spectra_values']
        self.data = np.concatenate(self.per_bird_data, axis=0)

    def getConeResponses(self, isWhitened=True):
        if hasattr(self, 'cone_responses'):
            return self.cone_responses
        pts = (self.observer.get_normalized_sensor_matrix()@self.data.T).T
        if isWhitened:
            log_usml = np.log10(pts)
            pts = log_usml - np.mean(log_usml, axis=0)
        return pts
    
    def getSpatialConeResponses(self, isWhitened=True):
        raise NotImplementedError("Spatial Cone Responses not implemented for Plumage Data Loader, because there are no images")


class NTIREDataLoader(DataLoader):
    def __init__(self, root_path, observer, offset=64):
        self.observer = observer
        self.dim = observer.dimension
        self.root = root_path
        self.data_range = [1, 900]

        self.offset = offset
        self.img_size = [self.offset*2, self.offset*2]

    @staticmethod
    def loadCube(path):
        """
        Load a spectral cube from Matlab HDF5 format .mat file
        :param path: Path of source file
        :return: spectral cube (cube) and bands (bands)
        """
        with h5py.File(path, 'r') as mat:
            cube = np.array(mat['cube']).T
            cube_bands = np.array(mat['bands']).squeeze()
        return cube, cube_bands
    
    def getithImage(self, i):
        """
        Load a spectral cube from Matlab HDF5 format .mat file
        input: index of the cube
        :return: spectral cube (cube) and bands (bands)
        """
        if i < self.data_range[0] or i > self.data_range[1]: 
            raise Exception("Outside of Range")
        filenames = [os.path.join(self.root, x) for x in sorted(os.listdir(self.root)) if os.path.splitext(x)[1] == '.mat']
        return NTIREDataLoader.loadCube(filenames[i])
    
    def getCroppedImage(self, img_idx, center= (482//2, 512//2), offset=64): 
        offsets = [[center[0] - offset, center[0] + offset], [center[1] - offset, center[1] + offset]]
        spectra_img = self.getithImage(img_idx)[0][offsets[0][0]:offsets[0][1], offsets[1][0]:offsets[1][1]].reshape(-1, 31)
        return spectra_img

    def getConeResponses(self, isWhitened=True):
        if hasattr(self, 'cone_responses'):
            return self.cone_responses
        LMSQ = []
        for i in tqdm(range(self.data_range[0], self.data_range[1]+1)):
            try:
                spectra_img = self.getCroppedImage(i, offset=self.offset)
            except:
                continue # image is corrupted most likely
            spectra_img_lmsq = (self.observer.get_normalized_sensor_matrix()@spectra_img.T).T

            if isWhitened: # Ruderman paper does per image whitening
                log_lmsq_img = np.log10(spectra_img_lmsq)
                pts = log_lmsq_img - np.mean(log_lmsq_img, axis=0) # .reshape(128, 128, 4) # don't want to reshape
            LMSQ += [pts]
        all_lmsq = np.stack(LMSQ, axis=0)
        self.cone_responses = all_lmsq
        return all_lmsq

    """
    Returns the 3x3 spatial grid of cone responses for each pixel in the image
    """
    def getSpatialConeResponses(self, isWhitened=True):
        if hasattr(self, 'spatial_pts'):
            return self.spatial_pts  
        if not hasattr(self, 'cone_responses'):
            self.getConeResponses(isWhitened=isWhitened)
        
        shifts = [(0, -1, -1, 0), (0, -1, 0, 0), (0, -1, 1, 0), (0, 0, -1, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 1, -1, 0), (0, 1, 0, 0), (0, 1, 1, 0)]
        shifted_LMSQ= [shift(self.cone_responses.reshape(-1, self.img_size[0], self.img_size[1], self.dim), s, order=0, prefilter=False) for s in shifts]
        LMSQ_334 = np.stack(shifted_LMSQ, axis=0)
        reordered = np.transpose(LMSQ_334, axes=[1, 2, 3, 0, 4])
        self.spatial_pts = reordered.reshape(-1, 9 * self.dim) # .reshape(-1, 128, 128, 9 * 4) # 36 dim vector
        return self.spatial_pts

class RudermanAnalysis: 
    def __init__(self, observer, data_loader):
        self.observer = observer
        self.data_loader = data_loader
        self.dim = observer.dimension
        self.n_comp_spatial = 3 * 3 * self.dim # 3x3 spatial grid

    def _pca_analysis(self, points, dim):
        pca = decomposition.PCA(n_components=dim)
        pca.fit_transform(points.reshape(-1, dim))
        print(pca.components_)
        print(pca.explained_variance_ratio_)
        return pca

    def doSpatialPCA(self):
        self.points = self.data_loader.getSpatialConeResponses()
        self.pca_spatial = self._pca_analysis(self.points, self.n_comp_spatial)

    def doConeResponsePCA(self):
        self.points = self.data_loader.getConeResponses()
        self.pca_cone = self._pca_analysis(self.points, self.dim)
        self.spectra_to_PCA = self.pca_cone.components_@(self.observer.get_normalized_sensor_matrix()) # 4x4 LMSQ to PCA

    def plotTransformedConeFunds(self):
        fig, ax = plt.subplots()
        colors = ['gray', 'blue', 'r', 'y'] # TODO: change colors
        for i in range(4):
            ax.plot(self.observer.wavelengths, self.spectra_to_PCA[i], label=f"PCA Component {i}", c=colors[i])
        ax.legend()
        plt.show()

    def plotPCA4D(self, viz, LMS_to_RGB, sample_rate=100):
        pca_comps = transformToChromaticity(self.pca_cone.components_)
        self.pca_cone.explained_variance_
        viz._getCoordBasis('pcacomps', pca_comps[1:], coordAlpha=1, colors=[[0, 0, 1], [1, 0, 0], [0, 0, 0]],radius=0.025/16 ) # first axis is luminance, which is nothing.

        lmsqs = self.data_loader.getConeResponses()
        lmsqs = lmsqs.reshape(-1, 4)[::sample_rate, :]
        rgbs = np.clip((LMS_to_RGB@(lmsqs[:, [0, 1, 3]].T)).T, 0, 1)

        chrom_pts = transformToChromaticity(lmsqs)
        viz.renderPointCloud("pca", chrom_pts, rgbs, radius=0.001)
        return 
    
    def plotPCA4DwUV(self, viz):
        pca_comps = transformToChromaticity(self.pca_cone.components_)
        viz._getCoordBasis('pcacomps',  pca_comps[1:], coordAlpha=1, colors=[[0, 0, 1], [1, 0, 0], [0, 0, 0]],radius=0.025/16 ) # first axis is luminance, which is nothing.
        
        lmsqs = self.data_loader.getConeResponses()
        rgbs = np.array(convert_refs_to_rgbs(self.data_loader.data[:, 10:], np.arange(400, 700 + 10, 10)))

        chrom_pts = transformToChromaticity(lmsqs)
        viz.renderPointCloud("pca", chrom_pts, rgbs, radius=0.001)
        return 
    
    def plotExplainedRatio(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], np.cumsum(self.pca_cone.explained_variance_ratio_))
        ax.set_xticks([1, 2, 3, 4])
        # ax.set_yticks(np.arange(0.7, 1.0, 0.05))
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        plt.show()

    
    def displaySpatialPCA(self, filename=None):
        img = self.pca_spatial.components_.reshape(-1, 3, 3, self.dim)
        fig, ax = plt.subplots(9, 4)
        fig_lqm, ax_lqm = plt.subplots(9, 4)
        for d in range(self.n_comp_spatial):
            i = d//4
            j = d % 4
            ax[i][j].imshow((128* (img[d]+1)).astype('int')[:, :, [3, 1, 0]])
            ax_lqm[i][j].imshow((128* (img[d]+1)).astype('int')[:, :, [3, 1, 2]])
        
        if filename is not None:
            fig.savefig(f"{filename}.png")
            fig_lqm.savefig(f"{filename}_lqm.png")
        else:
            plt.show()


# class TetraExplore:
#     def __init__(self, observer, data_loader, pca_obj) -> None:
#         self.pca = pca_obj
#         self.data_loader = data_loader
#         self.observer = observer
#         self.wavelengths = np.arange(400, 700 + 10, 10)

#     def projectToRGB(self, spectra_img):
#         return convert_refs_to_rgbs(spectra_img, self.wavelengths).reshape(128, 128, 3)
    
#     def saveIm(self, i, dict_name):
#         try:
#             spectra_img = self.data_loader.getithImage(i)[0]
#             shape = spectra_img.shape
#             spectra_img = spectra_img.reshape(-1, 31)
#         except:
#             return
#         PCA_Basis_im = (self.pca.spectra_to_PCA@(spectra_img.T)).T
#         rgb = self.projectToRGB(spectra_img)
#         score = PCA_Basis_im.max(axis=0)[3] - PCA_Basis_im.min(axis=0)[3]
#         PCA_Basis4 = PCA_Basis_im.reshape(shape[0], shape[1], 4)[:, :, 3]
#         plt.imshow(rgb.reshape(shape[0], shape[1], 3))
#         mynorm = plt.Normalize(vmin=-0.005, vmax=0.005)
#         plt.imshow(PCA_Basis4, cmap=plt.cm.coolwarm, alpha=0.7, interpolation='bilinear', norm=mynorm)
#         plt.colorbar()
#         plt.savefig(os.path.join(dict_name, f"pca_im_{i:03d}.png"))
#         plt.clf()  # Clear the figure for the next iteration
#         return score

#     def heatMapImages(self, topk=25):
#         dict = {}
#         for i in tqdm(range(1, 900+1, 1)):
#             score = self.saveIm(i, dict_name=f"./Results/PCA-Heatmaps/")
#             if score is not None:
#                 dict[i] = score
#         # ind = np.argpartition(scores, -topk)[-topk:]
#         k_keys_sorted_by_values = heapq.nlargest(topk, dict, key=dict.get)

#         print(k_keys_sorted_by_values)
#         print([dict[k] for k in k_keys_sorted_by_values])
#         for i in tqdm(k_keys_sorted_by_values):
#             score = self.saveIm(i, dict_name=f"./Results/top-PCA")
#         return k_keys_sorted_by_values

#     def displaySpectra(self, ind):
#         wavelengths = np.arange(400, 700 + 10, 10)
#         for img_ind in ind: 
#             try:
#                 spectra_img = self.data_loader.getithImage(img_ind)[0]
#                 shape = spectra_img.shape
#                 spectra_img = spectra_img.reshape(-1, 31)
#             except:
#                 return
#             PCA_Basis_im = (self.pca.spectra_to_PCA@(spectra_img.T)).T
#             PCA_Basis4 = PCA_Basis_im[:, 3]
#             top_ind = heapq.nlargest(5, range(len(PCA_Basis4)), PCA_Basis4.__getitem__)
#             rgb = self.projectToRGB(spectra_img)

#             fig = plt.figure()
#             subfigs = fig.subfigures(2, 1, hspace=0.07, height_ratios=[1.5, 1.])
#             axs0 = subfigs[0].subplots(1, 2)
#             subfigs[0].suptitle('RGB vs PCA Fourth Component Highlight')
#             axs0[0].imshow(rgb.reshape(shape[0], shape[1], 3))
#             axs0[0].axis("off")
#             mynorm = plt.Normalize(vmin=-0.005, vmax=0.005)
#             im1 = axs0[1].imshow(PCA_Basis4.reshape(shape[0], shape[1]), cmap=plt.cm.coolwarm, alpha=0.7, interpolation='bilinear', norm=mynorm)
#             axs0[1].axis('off')
#             divider = make_axes_locatable(axs0[1])
#             cax = divider.append_axes('right', size='5%', pad=0.05)
#             fig.colorbar(im1, cax=cax, orientation='vertical')

#             axs1 = subfigs[1].subplots(1, 5, sharey=True)
#             subfigs[1].suptitle('Top 5 Spectra in PCA Fourth Component')
#             # subfigs[1].supxlabel('Wavelength') # doesn't really work, overlaps
#             subfigs[1].supylabel('Reflectance')
#             for i, index in enumerate(top_ind):
#                 axs1[i].plot(wavelengths, spectra_img[index])
#             # plt.tight_layout()
#             plt.savefig(f"./Results/top-PCA/spectra_{img_ind:03d}.png", bbox_inches='tight')
#             plt.close()  # Clear the figure for the next iteration