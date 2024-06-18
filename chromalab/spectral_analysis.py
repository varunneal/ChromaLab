import os
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import heapq
import h5py

from sklearn import decomposition
from scipy.ndimage import shift
from tqdm import tqdm

from chromalab.spectra import Spectra, Illuminant


def loadCube(path):
    """
    Load a spectral cube from Matlab HDF5 format .mat file
    :param path: Path of souce file
    :return: spectral cube (cube) and bands (bands)
    """
    with h5py.File(path, 'r') as mat:
        cube = np.array(mat['cube']).T
        cube_bands = np.array(mat['bands']).squeeze()
    return cube, cube_bands


def getithImage(i):
    """
    Load a spectral cube from Matlab HDF5 format .mat file
    input: index of the cube
    :return: spectral cube (cube) and bands (bands)
    """
    if i < 1 or i > 900: 
        raise Exception("Outside of Range")
    data_dir = "/data/NTIRE_2022/"
    filenames = [os.path.join(data_dir, x) for x in sorted(os.listdir(data_dir)) if os.path.splitext(x)[1] == '.mat']
    return loadCube(filenames[i])



class PCAAnalysis:
    def __init__(self, observer):
        self.dim = observer.dimension
        self.n_comp_spatial = 3 * 3 * self.dim # 3x3 spatial grid
        self.observer = observer
        # Generate the points here (representation that can be used for normal and spatial)

    def _pca_analysis(self, points):
        pca = decomposition.PCA(n_components=self.dim)
        pca.fit_transform(points.reshape(-1, self.dim))
        self.spectra_to_PCA = pca.components_@self.cf.M # 4x4 LMSQ to PCA
        print(pca.components_)
        print(pca.explained_variance_ratio_)
        return pca

    def displaySpatialPCA(self, filename):
        img = self.pca_spatial.components_.reshape(-1, 3, 3, self.dim)
        fig, ax = plt.subplots(9, 4)
        fig_lqm, ax_lqm = plt.subplots(9, 4)
        for d in range(self.n_comp):
            i = d//4
            j = d % 4
            ax[i][j].imshow((128* (img[d]+1)).astype('int')[:, :, [3, 1, 0]])
            ax_lqm[i][j].imshow((128* (img[d]+1)).astype('int')[:, :, [3, 1, 2]])
        fig.savefig(f"{filename}.png")
        fig_lqm.savefig(f"{filename}_lqm.png")
        return

    def get_LMSQ_Spatial(self, points):
        shifts = [(0, -1, -1, 0), (0, -1, 0, 0), (0, -1, 1, 0), (0, 0, -1, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 1, -1, 0), (0, 1, 0, 0), (0, 1, 1, 0)]
        shifted_LMSQ= [shift(points, s, order=0, prefilter=False) for s in shifts]
        LMSQ_334 = np.stack(shifted_LMSQ, axis=0)
        reordered = np.transpose(LMSQ_334, axes=[1, 2, 3, 0, 4])
        self.spatial_pts =  reordered.reshape(-1, 128, 128, 9 * 4) # 36 dim vector

    def get_LMSQ_Responses(self):
        LMSQ =[]
        center = (482//2, 512//2)
        offsets = [[center[0] - 64, center[0] + 64], [center[1] - 64, center[1] + 64]] # center crop image (they did this in the paper)
        for i in tqdm(range(1, 900+1)):
            try:
                spectra_img=getithImage(i)[0][offsets[0][0]:offsets[0][1], offsets[1][0]:offsets[1][1]].reshape(-1, 31)
            except:
                continue
            LMSQ += [((self.observer.get_normalized_sensor_matrix()@spectra_img.T).T)]
        pts = np.stack(LMSQ, axis=0)
        return pts

    def get_whitened_LMSQ_Responses(self):
        if hasattr(self, 'points'):
            return
        LMSQ =[]
        center = (482//2, 512//2)
        offsets = [[center[0] - 64, center[0] + 64], [center[1] - 64, center[1] + 64]] # center crop image (they did this in the paper)
        for i in tqdm(range(1, 900+1)):
            try:
                spectra_img=getithImage(i)[0][offsets[0][0]:offsets[0][1], offsets[1][0]:offsets[1][1]].reshape(-1, 31)
            except:
                continue
            # colour_d65 = Illuminant.get('D65')
            # idx1 = (self.cf.minwave-300)//5
            # idx2 = (self.cf.maxwave + 5 - 300)//5
            # d65 = colour_d65.values[idx1:idx2:2] # / max(colour_d65.values[idx1:idx2:2])
            # LMSQ_vals = np.log10((self.cf.M@(spectra_img * d65).T).T) # change these lines
            LMSQ_vals = np.log10((self.observer.observe(spectra_img)).T)
            whitened_vals = LMSQ_vals - np.mean(LMSQ_vals, axis=0)
            whitened_vals = whitened_vals.reshape(128, 128, 4)
            LMSQ += [whitened_vals]

        pts = np.stack(LMSQ, axis=0)
        self.points = pts
    
    def analyzeConeFund(self):
        self.get_whitened_LMSQ_Responses()
        self.pca_cone = self.pca_analysis(self.points)
        
    def analyzeSpatial(self, spatial_pca_filename):
        self.get_LMSQ_Spatial(self.points)
        self.pca_spatial = self.pca_analysis(self.spatial_pts)
        self.displaySpatialPCA(spatial_pca_filename)
     
    def analyzeAll(self, spatial_pca_filename):
        self.analyzeConeFund()
        self.analyzeSpatial(spatial_pca_filename)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class TetraExplore:
    def __init__(self, pca_obj) -> None:
        self.pca = pca_obj
    
    def saveIm(self, i, dict_name):
        try:
            spectra_img = getithImage(i)[0]
            shape = spectra_img.shape
            spectra_img = spectra_img.reshape(-1, 31)
        except:
            return
        PCA_Basis_im = (self.pca.spectra_to_PCA@(spectra_img.T)).T
        rgb = projectToRGB(spectra_img)
        score = PCA_Basis_im.max(axis=0)[3] - PCA_Basis_im.min(axis=0)[3]
        PCA_Basis4 = PCA_Basis_im.reshape(shape[0], shape[1], 4)[:, :, 3]
        plt.imshow(rgb.reshape(shape[0], shape[1], 3))
        mynorm = plt.Normalize(vmin=-0.005, vmax=0.005)
        plt.imshow(PCA_Basis4, cmap=plt.cm.coolwarm, alpha=0.7, interpolation='bilinear', norm=mynorm)
        plt.colorbar()
        plt.savefig(os.path.join(dict_name, f"pca_im_{i:03d}.png"))
        plt.clf()  # Clear the figure for the next iteration
        return score

    def heatMapImages(self, topk=25):
        dict = {}
        for i in tqdm(range(1, 900+1, 1)):
            score = self.saveIm(i, dict_name=f"./Results/PCA-Heatmaps/")
            if score is not None:
                dict[i] = score
        # ind = np.argpartition(scores, -topk)[-topk:]
        k_keys_sorted_by_values = heapq.nlargest(topk, dict, key=dict.get)

        print(k_keys_sorted_by_values)
        print([dict[k] for k in k_keys_sorted_by_values])
        for i in tqdm(k_keys_sorted_by_values):
            score = self.saveIm(i, dict_name=f"./Results/top-PCA")
        return k_keys_sorted_by_values

    def displaySpectra(self, ind):
        wavelengths = np.arange(400, 700 + 10, 10)
        for img_ind in ind: 
            try:
                spectra_img = getithImage(img_ind)[0]
                shape = spectra_img.shape
                spectra_img = spectra_img.reshape(-1, 31)
            except:
                return
            PCA_Basis_im = (self.pca.spectra_to_PCA@(spectra_img.T)).T
            PCA_Basis4 = PCA_Basis_im[:, 3]
            top_ind = heapq.nlargest(5, range(len(PCA_Basis4)), PCA_Basis4.__getitem__)
            rgb = projectToRGB(spectra_img)

            fig = plt.figure()
            subfigs = fig.subfigures(2, 1, hspace=0.07, height_ratios=[1.5, 1.])
            axs0 = subfigs[0].subplots(1, 2)
            subfigs[0].suptitle('RGB vs PCA Fourth Component Highlight')
            axs0[0].imshow(rgb.reshape(shape[0], shape[1], 3))
            axs0[0].axis("off")
            mynorm = plt.Normalize(vmin=-0.005, vmax=0.005)
            im1 = axs0[1].imshow(PCA_Basis4.reshape(shape[0], shape[1]), cmap=plt.cm.coolwarm, alpha=0.7, interpolation='bilinear', norm=mynorm)
            axs0[1].axis('off')
            divider = make_axes_locatable(axs0[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax, orientation='vertical')

            axs1 = subfigs[1].subplots(1, 5, sharey=True)
            subfigs[1].suptitle('Top 5 Spectra in PCA Fourth Component')
            # subfigs[1].supxlabel('Wavelength') # doesn't really work, overlaps
            subfigs[1].supylabel('Reflectance')
            for i, index in enumerate(top_ind):
                axs1[i].plot(wavelengths, spectra_img[index])
            # plt.tight_layout()
            plt.savefig(f"./Results/top-PCA/spectra_{img_ind:03d}.png", bbox_inches='tight')
            plt.close()  # Clear the figure for the next iteration

            
            

        
        

