from itertools import combinations
from functools import reduce
from tqdm import tqdm

import numpy as np
import glm
import matplotlib.pyplot as plt

from .observer import Observer, getHeringMatrix
from .spectra import Spectra, convert_refs_to_spectras, Illuminant


def getCylinderTransform(endpoints):
    a = endpoints[1]-endpoints[0]
    a = glm.vec3(a[0], a[1], a[2])
    b = glm.vec3(0, 0, 1)
    
    mat = glm.mat4()
    # translate
    mat = glm.translate(mat, glm.vec3(endpoints[0][0], endpoints[0][1], endpoints[0][2]))
    # rotate
    v = glm.cross(b, a)
    if v!= glm.vec3(0, 0, 0):
        angle = glm.acos(glm.dot(b, a) / (glm.length(b) * glm.length(a)))
        mat = glm.rotate(mat, angle, v)

    # scale
    scale_factor = glm.length(a)
    mat = glm.scale(mat, glm.vec3(scale_factor, scale_factor, scale_factor))

    return mat

class MaxBasis:
    dim4SampleConst = 10
    dim3SampleConst = 2
    
    def __init__(self, observer, verbose=False) -> None:
        self.verbose = verbose
        self.observer = observer
        self.wavelengths = observer.wavelengths
        self.matrix = observer.get_normalized_sensor_matrix()
        self.dimension = observer.dimension
        self.step_size = self.observer.wavelengths[1] - self.observer.wavelengths[0]
        self.dim_sample_const = self.dim4SampleConst if self.dimension == 4 else self.dim3SampleConst

        self.HMatrix = getHeringMatrix(observer.dimension)

        self.__getMaximalBasis()
        # self.__findConeToBasisTransform(isReverse=True)


    def __computeVolume(self, wavelengths):
        # wavelengths = [matrix.wavelengths[idx] for idx in indices]
        transitions = self.getCutpointTransitions(wavelengths)
        cone_vals = np.array([np.dot(self.matrix, Spectra.from_transitions(x, 1 if i == 0 else 0, self.wavelengths).data) for i, x in enumerate(transitions)])
        vol = np.abs(np.linalg.det(cone_vals))
        return vol

    def __findMaximalCMF(self, isReverse=True):
        sortedCutpoints = self.cutpoints[:self.dimension - 1]
        sortedCutpoints.sort()
        transitions = self.getCutpointTransitions(sortedCutpoints)
        if isReverse:
            refs = np.array([Spectra.from_transitions( x, 1 if i == 0 else 0, self.wavelengths).data for i, x in enumerate(transitions)])[::-1]
        else:
            refs = np.array([Spectra.from_transitions( x, 1 if i == 0 else 0, self.wavelengths).data for i, x in enumerate(transitions)])
        self.refs = refs
        self.cone_to_maxbasis = np.linalg.inv(np.dot(self.matrix, refs.T))
        self.maximal_matrix = np.dot(self.cone_to_maxbasis, self.matrix)

        self.maximal_sensors = []
        for i in range(self.dimension):
            spectra = np.concatenate([self.observer.wavelengths[:, np.newaxis], self.maximal_matrix[i][:, np.newaxis]], axis=1)
            self.maximal_sensors +=[Spectra(spectra, self.observer.wavelengths, normalized=False)] # false to stop clipping from 0 to 1
        self.maximal_observer = Observer(self.maximal_sensors, verbose=self.verbose) # erased self.observer.illuminant.
        return self.maximal_sensors, self.maximal_observer

    
    # def __findConeToBasisTransform(self, isReverse=True):
        # sortedCutpoints = self.cutpoints[:self.dimension - 1]
        # sortedCutpoints.sort()
        # transitions = self.getCutpointTransitions(sortedCutpoints)
        # if isReverse:
        #     refs = np.array([Spectra.from_transitions( x, 1 if i == 0 else 0, self.wavelengths).data for i, x in enumerate(transitions)])[::-1]
        # else:
        #     refs = np.array([Spectra.from_transitions(x, 1 if i == 0 else 0, self.wavelengths).data for i, x in enumerate(transitions)])
        # self.cone_to_maxbasis = np.linalg.inv(np.dot(self.matrix, refs.T))


        # new function
        # self.cone_to_maxbasis = np.linalg.inv(self.maximal_matrix@self.matrix)


    def __findMaxCutpoints(self, rng=None):
        if self.dimension == 2:
            X = np.arange(self.observer.wavelengths[0] + self.step_size, self.observer.wavelengths[-1]  - self.step_size, self.step_size)
            Xidx = np.meshgrid(X)[0]
            Zidx = np.zeros_like(Xidx, dtype=float)
            
            for i in tqdm(range(len(X)), disable=not self.verbose ):
                wavelength = [Xidx[i]]
                Zidx[i] = self.__computeVolume(wavelength)
            self.listvol = [Xidx, Zidx]
            maxvol = reduce(max, Zidx.flatten())
            idxs = np.where(Zidx == maxvol)
            self.cutpoints = [Xidx[idxs][0], Zidx[idxs][0]]
            return self.cutpoints
        
        elif self.dimension == 3:
            if not rng:
                X = np.arange(self.observer.wavelengths[0] + self.step_size, self.observer.wavelengths[-1]  - self.step_size, self.step_size)
                Y = np.arange(self.observer.wavelengths[0] + self.step_size, self.observer.wavelengths[-1]  - self.step_size, self.step_size)
            else: 
                X = np.arange(rng[0][0], rng[0][1], self.step_size)
                Y = np.arange(rng[1][0], rng[1][1], self.step_size)
            Xidx, Yidx = np.meshgrid(X, Y, indexing='ij')
            Zidx = np.zeros_like(Xidx, dtype=float)
            for i in tqdm(range(len(X)), disable=not self.verbose):
                for j in range(len(Y)):
                    if i <=j:
                        wavelengths = [Xidx[i, j], Yidx[i, j]];
                        wavelengths.sort()
                        Zidx[i, j] = self.__computeVolume(wavelengths)
            self.listvol = [Xidx, Yidx, Zidx]
            maxvol = reduce(max, Zidx.flatten())
            idxs = np.where(Zidx == maxvol)
            self.cutpoints = [Xidx[idxs][0], Yidx[idxs][0], Zidx[idxs][0]]
            return self.cutpoints
        elif self.dimension == 4:
            if not rng:
                X = np.arange(self.observer.wavelengths[0] + self.step_size, self.observer.wavelengths[-1]  - self.step_size, self.step_size)
                Y = np.arange(self.observer.wavelengths[0] + self.step_size, self.observer.wavelengths[-1]  - self.step_size, self.step_size)
                W = np.arange(self.observer.wavelengths[0] + self.step_size, self.observer.wavelengths[-1]  - self.step_size, self.step_size)
            else:
                X = np.arange(rng[0][0], rng[0][1], self.step_size)
                Y = np.arange(rng[1][0], rng[1][1], self.step_size)
                W = np.arange(rng[2][0], rng[2][1], self.step_size)
            Xidx, Yidx, Widx = np.meshgrid(X, Y, W, indexing='ij')

            Zidx = np.zeros_like(Xidx, dtype=float)
            for i in tqdm(range(len(X))):
                for j in range(len(Y)):
                    for k in range(len(W)):
                            if i <=j and j <= k :
                                wavelengths = [Xidx[i, j, k], Yidx[i, j, k], Widx[i, j, k]];
                                wavelengths.sort()
                                Zidx[i, j, k] = self.__computeVolume(wavelengths)
            self.listvol = [Xidx, Yidx, Widx, Zidx]
            maxvol = reduce(max, Zidx.flatten())
            idxs = np.where(Zidx == maxvol)
            self.idxs =[x.tolist()[0] for x in idxs]
            self.cutpoints = [Xidx[idxs][0], Yidx[idxs][0], Widx[idxs][0], Zidx[idxs][0]]
            return self.cutpoints
        else:
            raise NotImplementedError
    
    def __getMaximalBasis(self,rng=None):
        range = []
        if self.step_size < 2 and self.dimension > 2 : # find a range to do fine-grained search to narrow down brute force
            rangbd = int(self.dim_sample_const * 2)
            coarse_wavelengths = np.arange(self.observer.wavelengths[0] + self.step_size, self.observer.wavelengths[-1]  - self.step_size, self.dim_sample_const)
            coarse_sensors = [s.interpolate_values(coarse_wavelengths) for s in self.observer.sensors]
            coarseObserver = Observer(coarse_sensors, self.observer.illuminant)
            coarseMaxBasis = MaxBasis(coarseObserver, verbose=self.verbose)
            cutpoints = coarseMaxBasis.get_cutpoints()
            range = [[x - rangbd, x + rangbd] for x in cutpoints[:self.dimension-1]]

        self.__findMaxCutpoints(range)
        self.__findMaximalCMF(isReverse=False)

    def get_max_basis_observer(self):
        return self.maximal_observer

    def get_cone_to_maxbasis_transform(self):
        return self.cone_to_maxbasis
    
    def get_cutpoints(self):
        return self.cutpoints
    
    def get_cmf(self):
        return self.maximal_sensors
    
    def getCutpointTransitions(self, wavelengths):
        transitions = [[wavelengths[0]], [wavelengths[len(wavelengths)-1]]]
        transitions += [[wavelengths[i], wavelengths[i+1]] for i in range(len(wavelengths)-1)]
        transitions.sort()
        return transitions
    
    def getMatrixOrientationQUp(self):
        length = 1
        basisLMSQ = np.array([[0, 0, 1, 0]]) * length # Q cone
        basisLMSQ = ((self.HMatrix@self.get_cone_to_maxbasis_transform())@basisLMSQ.T).T[:, 1:][0]
        # matrix = np.linalg.inv(getCylinderTransform([np.array([0, 0, 0]), basisLMSQ/np.linalg.norm(basisLMSQ)])) # go from Q to the basis [0, 0, 1]
        matrix = np.array(getCylinderTransform([np.array([0, 0, 0]), basisLMSQ/np.linalg.norm(basisLMSQ)]))
        return matrix[:3, :3]
    
    def getRotMatQUp(self):
        Q_vec = [self.cone_to_maxbasis @np.array([0, 0, 1, 0]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0])] # Q direction
        def gram_schmidt(vectors):
            basis = []
            for v in vectors:
                w = v - np.sum( np.dot(v,b)*b  for b in basis )
                if (w > 1e-10).any():  
                    basis.append(w/np.linalg.norm(w))
            return np.array(basis)
        A = gram_schmidt(Q_vec)
        return A

    def getDiscreteRepresentation(self, reverse=False):
        transitions = self.getCutpointTransitions(self.cutpoints[:self.dimension-1])
        start_vals = [1 if i == 0 else 0 for i, x in enumerate(transitions)]
        allcombos = [[]]
        alllines = []
        allstart = [[0]] # black starting value
        for i in range(self.dimension):
            alllines += list(combinations(range(self.dimension), i + 1))
            allcombos += [[elem for lst in x for elem in lst] for x in list(combinations(transitions, i + 1))]
            allstart += [list(x) for x in list(combinations(start_vals, i + 1))]
        final_start =  [max(x) for x in allstart]
        final_combos = []
        for x in allcombos:
            lst_elems, counts = np.unique(x, return_counts=True)
            removeIdx = []
            lst_elems = list(lst_elems)
            num_elems = len(lst_elems)
            for i, cnt in enumerate(reversed(list(counts))): 
                if cnt > 1:
                    del lst_elems[num_elems - i -1]
            lst_elems.sort()
            final_combos += [lst_elems]
        lines = []
        for i, x in enumerate(alllines): 
            if len(x) <=1:
                lines += [[0, x[0] + 1]] # connected to black
            else: # > 1
                madeupof = list(combinations(x, len(x)-1))
                lines += [[alllines.index(elem) + 1, i + 1] for elem in madeupof] # connected to each elem
        refs = [Spectra.from_transitions( x, final_start[i], self.wavelengths) for i, x in enumerate(final_combos)]
        if reverse:
            points = np.array([self.maximal_matrix[::-1] @ ref.data for ref in refs])
        else:
            points = np.array([self.maximal_matrix @ ref.data for ref in refs])
        rgbs = np.array([s.to_rgb(illuminant=Illuminant.get("E")) for s in refs])
        return refs, points, rgbs, lines