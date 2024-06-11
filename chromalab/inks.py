import heapq
from itertools import combinations, chain
from time import time, perf_counter
from typing import List, Union, Optional, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from itertools import product, islice, combinations


from typing import Tuple, Iterable, List
from tqdm import tqdm
from itertools import product

from collections import defaultdict
import math

from .spectra import Spectra
from .observer import Observer, getHeringMatrix, transformToChromaticity
from .maxbasis import MaxBasis

from sklearn.decomposition import PCA, TruncatedSVD
from scipy.special import comb
from scipy.spatial import ConvexHull


class Pigment(Spectra):
    # TODO: reformat with kwargs
    def __init__(self, array: Optional[Union[Spectra, npt.NDArray]] = None,
                 k: Optional[npt.NDArray] = None,
                 s: Optional[npt.NDArray] = None,
                 wavelengths: Optional[npt.NDArray] = None,
                 data: Optional[npt.NDArray] = None, **kwargs):
        """
        Either pass in @param reflectance or pass in
        @param k and @param s.

        k and s are stored as spectra rather than NDArrays.
        """
        if k is not None and s is not None and wavelengths is not None:
            # compute reflectance from k & s
            if not k.shape == s.shape or not k.shape == wavelengths.shape:
                raise ValueError("Coefficients k and s and wavelengths must be same shape.")

            r = 1 + (k / s) - np.sqrt(np.square(k / s) + (2 * k / s))
            super().__init__(wavelengths=wavelengths, data=r, **kwargs)

        elif array is not None:
            if isinstance(array, Spectra):
                super().__init__(**array.__dict__, **kwargs)
            else:
                super().__init__(array=array, wavelengths=wavelengths, data=data, **kwargs)
            k, s = self.compute_k_s()
        elif data is not None:
            super().__init__(wavelengths=wavelengths, data=data, **kwargs)

        else:
            raise ValueError("Must either specify reflectance or k and s coefficients and wavelengths.")

        self.k, self.s = k, s

    def compute_k_s(self) -> Tuple[npt.NDArray, npt.NDArray]:
        # Walowit · 1987 specifies this least squares method
        # todo: GJK method as per Centore • 2015
        array = np.clip(self.array(), 1e-4, 1)
        k, s = [], []
        for wavelength, r in array:
            k_over_s = (1 - r) * (1 - r) / (2 * r)
            A = np.array([[-1, k_over_s], [1, 1]])
            b = np.array([0, 1])

            AtA_inv = np.linalg.inv(np.dot(A.T, A))
            Atb = np.dot(A.T, b)

            _k, _s = np.dot(AtA_inv, Atb)
            k.append(_k)
            s.append(_s)

        return np.clip(np.array(k), 0, 1), np.clip(np.array(s), 0, 1)

    def get_k_s(self) -> Tuple[npt.NDArray, npt.NDArray]:
        # todo: pass in wavelength list for interpolation/sampling consistency with mixing
        return self.k, self.s


def get_metamers(points, target, threshold=1e-2, axis=2):
    # ok the idea here is to find a point in points
    # that is a confusion point wrt target
    metamers = []
    for idx, point in enumerate(points):
        # print(point)
        metamer_closeness = math.sqrt(
                sum(
                    [
                (point[i] - target[i]) ** 2
            for i in range(len(point)) if i != axis])
        )
        if metamer_closeness < threshold:
            metamers.append((abs(target[axis] - point[axis]), idx))
    metamers.sort(reverse=True)
    return metamers


def lsh_buckets(points, ignored_axis=2):
    # Implementation of Locally Sensitive Hashing
    print(points.shape)
    n_dimensions = points.shape[1]

    weights = np.zeros(n_dimensions)
    for i in range(n_dimensions):
        if i != ignored_axis:
            weights[i] = 10 ** (2 * (i + 1))

    weights = np.zeros(n_dimensions)
    adjustments = []
    for i in range(n_dimensions):
        if i != ignored_axis:
            weights[i] = 10 ** (2 * (i + 1))
            exp_base = 10 ** (2 * (i + 1)) // 2
            adjustments.extend([exp_base, -exp_base])

    # Calculate base hash values excluding the ignored axis
    base_hashes = np.dot(points, weights)

    # Apply adjustments and calculate hash values
    hash_values = np.array([np.floor(base_hashes + adjustment).astype(int) for adjustment in adjustments])

    return hash_values


def bucket_points(points: npt.NDArray , axis=2):
    # disjointed buckets
    buckets = defaultdict(list)
    N, d = points.shape

    prec = 0.005
    # 8 is large enough for prec = 0.005:
    # 8 > log_2 (1 / 0.005)
    weights = (2 ** (8 * np.arange(0, d)))
    weights[axis] = 0

    values = points // prec
    keys = values @ weights
    for i, (key, point) in enumerate(zip(keys, values)):
        buckets[key].append((point / 2, i))  # can replace 2 with 0.01 // prec

    return {k: v for k, v in buckets.items() if len(v) > 1}


def sort_buckets(buckets, axis=2) -> List:
    dist_buckets = []

    for metamers in buckets.values():
        if len(metamers) <= 1:
            continue

        axis_values = [metamer[0][axis] for metamer in metamers]

        min_val = min(axis_values)
        max_val = max(axis_values)

        distance = max_val - min_val

        min_index = axis_values.index(min_val)
        max_index = axis_values.index(max_val)
        best_indices = (metamers[min_index][1], metamers[max_index][1])

        dist_buckets.append((distance, best_indices))

    return sorted(dist_buckets, reverse=True)


def max_dist(buckets, axis=2):
    return sort_buckets(buckets, axis=axis)[0]


def k_s_from_data(data: npt.NDArray):
    array = np.clip(data, 1e-4, 1)
    k, s = [], []
    k_over_s = (1 - array) ** 2 / (2 * array)
    b = np.array([0, 1])

    for f in k_over_s:
        A = np.array([[-1, f], [1, 1]])

        # should just be able to call np.linalg.inverse
        AtA_inv = np.linalg.inv(np.dot(A.T, A))
        Atb = np.dot(A.T, b)
        _k, _s = np.dot(AtA_inv, Atb)

        k.append(_k)
        s.append(_s)

    return np.clip(np.array(k), 0, 1), np.clip(np.array(s), 0, 1)


def data_from_k_s(k, s):
    return 1 + (k / s) - np.sqrt(np.square(k / s) + (2 * k / s))


def k_s_from_pigments(pigments):
    k_s_pairs = [k_s_from_data(pigment.data) for pigment in pigments]
    k_matrix, s_matrix = np.array([k for k, _ in k_s_pairs]), np.array([s for _, s in k_s_pairs])

    return k_matrix.T, s_matrix.T


def km_mix(pigments, concentrations=None):
    K_matrix, S_matrix = k_s_from_pigments(pigments)
    wavelengths = pigments[0].wavelengths

    if not concentrations:
        concentrations = np.ones(len(pigments)) / len(pigments)

    K_mix = K_matrix @ concentrations
    S_mix = S_matrix @ concentrations

    return Spectra(data=data_from_k_s(K_mix, S_mix), wavelengths=wavelengths)


def load_neugebauer(inks, paper, n=50):
    num_inks = len(inks)
    primaries_dict = {'0' * num_inks: paper}

    for i in range(1, 2 ** num_inks):
        binary_str = format(i, f'0{num_inks}b')
        inks_to_mix = []

        for j, bit in enumerate(binary_str):
            if bit == '1':
                inks_to_mix.append(inks[j])

        if binary_str not in primaries_dict:  # should not need if statement
            mixed_ink = inks_to_mix[0]
            if len(inks_to_mix) > 1:
                mixed_ink = km_mix(inks_to_mix)
            primaries_dict[binary_str] = mixed_ink
    return Neugebauer(primaries_dict, n=n)


def observe_spectra(data, observer, illuminant):
    return np.divide(np.matmul(observer, (data * illuminant).T).squeeze(), np.matmul(observer, illuminant.T))


def observe_spectras(spectras: npt.NDArray, observer: npt.NDArray, illuminant: npt.NDArray) -> npt.NDArray:
    numerator = np.matmul(observer, (spectras * illuminant).T)
    denominator = np.matmul(observer, illuminant.T)
    result = np.divide(numerator, denominator[:, np.newaxis])
    return result.T

def find_best_n(primaries_dict, percentages: npt.NDArray, actual: Spectra):
    # Find best n array for a particular sample
    wavelengths = actual.wavelengths

    best_n = None
    best_error = float('inf')

    candidates = np.logspace(-2, 2, num=100, base=10)
    # print(candidates)
    # inefficient but idc
    for n in candidates:
        neug_n = Neugebauer(primaries_dict, n=n)
        result = neug_n.mix(percentages).reshape(-1)
        error = np.sum(np.square((actual.data - result)))
        if error < best_error:
            best_n = n
            best_error = error

    return best_n


def find_best_ns(primaries_dict, samples_dict):
    # samples dict will need to be (Tuple, Spectra) pairs
    best_ns = []

    for percentage, actual in samples_dict.items():
        best_n = find_best_n(primaries_dict, np.array(percentage), actual)
        best_ns.append(best_n)

    return best_ns


class InkLibrary:
    """
    InkLibrary is a class equipped with a method for "fast ink gamut search". It uses PCA to identify
    the best k-ink set from a larger library.
    """
    def __init__(self, library: Dict[str, Spectra], paper: Spectra):
        self.names = list(library.keys())
        spectras = list(library.values())
        self.wavelengths = spectras[0].wavelengths
        self.spectras = np.array([s.data for s in spectras])
        self.K = len(self.names)
        for s in spectras:
            assert np.array_equal(s.wavelengths, self.wavelengths)
        assert np.array_equal(self.wavelengths, paper.wavelengths)
        self.paper = paper.data

    def distance_search(self, observe: Union[Observer, npt.NDArray],
                          illuminant: Union[Spectra, npt.NDArray], top=100, k=None, stepsize=0.1):
        # slow unoptimized method, finds max q distance along point cloud
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(self.wavelengths)
        if k is None:
            print(type(observe))
            k = observe.shape[0]

        top_scores = []
        for inkset_idx in tqdm(combinations(range(self.K), k), total=comb(self.K, k), desc="finding best inkset"):
            names = [self.names[i] for i in inkset_idx]

            spectras = [Spectra(wavelengths=self.wavelengths, data=self.spectras[idx]) for idx in inkset_idx]

            gamut = InkGamut(spectras, Spectra(wavelengths=self.wavelengths, data=self.paper), illuminant)
            score = gamut.get_width(observe, stepsize=stepsize, verbose=False)
            if len(top_scores) < top:
                heapq.heappush(top_scores, (score, names))
            else:
                if score > top_scores[0][0]:
                    heapq.heapreplace(top_scores, (score, names))
        return sorted(top_scores, reverse=True)



    def convex_hull_search(self, observe: Union[Observer, npt.NDArray],
                          illuminant: Union[Spectra, npt.NDArray], top=100, k=None):
        # super efficient way to find best k-ink subset of large K ink library
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(self.wavelengths)
        if isinstance(illuminant, Spectra):
            illuminant = illuminant.interpolate_values(self.wavelengths).data
        if k is None:
            k = observe.shape[0]

        km_cache = {}

        total_iterations = sum(comb(self.K, i) for i in range(2, k + 1))

        ks_cache = []
        for ink in self.spectras:
            ks_cache.append(np.stack(k_s_from_data(ink)))

        with tqdm(total=total_iterations, desc="loading km cache") as pbar:
            for i in range(2, k + 1):
                concentrations = np.ones(i) / i
                for subset in combinations(range(self.K), i):
                    inks_to_mix = [ks_cache[idx] for idx in subset]
                    ks_batch = np.stack(inks_to_mix, axis=2)
                    ks_mix = ks_batch @ concentrations
                    data = data_from_k_s(ks_mix[0], ks_mix[1])
                    km_cache[subset] = data.astype(np.float16)
                    pbar.update(1)

        del ks_cache

        denominator = np.matmul(observe, illuminant.T)[:, np.newaxis]

        top_scores = []
        for inkset_idx in tqdm(combinations(range(self.K), k), total=comb(self.K, k), desc="finding best inkset"):
            names = [self.names[i] for i in inkset_idx]

            primaries_array = [self.paper]
            primaries_array.extend([self.spectras[idx] for idx in inkset_idx])
            primaries_array.extend(
                [km_cache[subset] for i in range(2, k + 1) for subset in combinations(inkset_idx, i)])

            primaries_array = np.array(primaries_array)

            numerator = np.matmul(observe, (primaries_array * illuminant).T)
            observe_mix = np.divide(numerator, denominator) # 4 x 16
            vol = ConvexHull(observe_mix.T).volume

            if len(top_scores) < top:
                heapq.heappush(top_scores, (vol, names))
            else:
                if vol > top_scores[0][0]:
                    heapq.heapreplace(top_scores, (vol, names))

        return  sorted(top_scores, reverse=True)



    def cached_pca_search(self, observe: Union[Observer, npt.NDArray],
                          illuminant: Union[Spectra, npt.NDArray], top=50, k=None):
        # super efficient way to find best k-ink subset of large K ink library
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(self.wavelengths)
        if isinstance(illuminant, Spectra):
            illuminant = illuminant.interpolate_values(self.wavelengths).data
        if k is None:
            k = observe.shape[0]

        km_cache = {}
        # Populate cache
        total_iterations = sum(comb(self.K, i) for i in range(2, k + 1))

        top_scores = []

        with tqdm(total=total_iterations, desc="loading km cache") as pbar:
            ks_cache = []
            for ink in self.spectras:
                ks_cache.append(np.stack(k_s_from_data(ink)))

            for i in range(2, k + 1):
                concentrations = np.ones(i) / i
                for subset in combinations(range(self.K), i):
                    inks_to_mix = [ks_cache[idx] for idx in subset]
                    ks_batch = np.stack(inks_to_mix, axis=2)
                    ks_mix = ks_batch @ concentrations
                    data = data_from_k_s(ks_mix[0], ks_mix[1])
                    km_cache[subset] = data.astype(np.float16)
                    pbar.update(1)

            del ks_cache

        weights_array = []
        for i in range(k+1):
            for subset in combinations(range(k), i):
                binary = [0] * k
                for index in subset:
                    binary[index] = 1
                weights_array.append(binary)
        weights_array = np.array(weights_array)

        # nothing
        top_scores = []

        stepsize = 0.2
        values = np.arange(0, 1 + stepsize, stepsize)
        mesh = np.meshgrid(*([values] * k))
        grid = np.stack(mesh, axis=-1).reshape(-1, k)

        w_p = ((weights_array * grid[:, np.newaxis, :]) +
               (1 - weights_array) * (1 - grid[:, np.newaxis, :]))
        w_p_prod = np.prod(w_p, axis=2, keepdims=True)

        pca = PCA(n_components=observe.shape[0])
        tsvd = TruncatedSVD(n_components=observe.shape[0], n_iter=5, random_state=42)

        denominator = np.matmul(observe, illuminant.T)[:, np.newaxis]

        inkset_iteration_tt = 0
        # Find best inkset
        iters = 0
        for inkset_idx in tqdm(combinations(range(self.K), k), total=comb(self.K, k), desc="finding best inkset"):

            names = [self.names[i] for i in inkset_idx]

            primaries_array = [self.paper]
            primaries_array.extend([self.spectras[idx] for idx in inkset_idx])
            primaries_array.extend(
                [km_cache[subset] for i in range(2, k + 1) for subset in combinations(inkset_idx, i)])

            primaries_array = np.array(primaries_array)
            data_array = np.power(primaries_array, 1.0 / 50)



            mix = np.power(np.matmul(w_p_prod.transpose(0, 2, 1), data_array), 50).squeeze(axis=1)

            numerator = np.matmul(observe, (mix * illuminant).T)
            observe_mix = np.divide(numerator, denominator)
            pca.fit(observe_mix)
            score = np.sqrt(pca.explained_variance_)[-1]


            if len(top_scores) < top:
                heapq.heappush(top_scores, (score, names))
            else:
                if score > top_scores[0][0]:
                    heapq.heapreplace(top_scores, (score, names))

            iters += 1
            if iters > 1000:
                print("pca iteration", inkset_iteration_tt / iters, "seconds")

                return

        return sorted(top_scores, reverse=True)


class FastNeugebauer:
    def __init__(self, weights_array, data_array, num_inks):
        self.weights_array = weights_array
        self.data_array = np.power(data_array, 1.0 / 50)
        self.num_inks = num_inks


    def batch_mix(self, percentages: npt.NDArray) -> npt.NDArray:
        w_p = ((self.weights_array * percentages[:, np.newaxis, :]) +
               (1 - self.weights_array) * (1 - percentages[:, np.newaxis, :]))
        w_p_prod = np.prod(w_p, axis=2, keepdims=True)

        result = np.power(np.matmul(w_p_prod.transpose(0, 2, 1), self.data_array), 50).squeeze(axis=1)

        return result

    def batch_observe(self, percentages: npt.NDArray, observer: npt.NDArray, illuminant: npt.NDArray):
        spectras = self.batch_mix(percentages)
        numerator = np.matmul(observer, (spectras * illuminant).T)
        denominator = np.matmul(observer, illuminant.T)
        result = np.divide(numerator, denominator[:, np.newaxis])
        return result.T

    def get_pca_size(self, grid, observe: npt.NDArray, illuminant: npt.NDArray):
        stimulus = self.batch_observe(grid, observe, illuminant)

        pca = PCA(n_components=observe.shape[0])

        pca.fit(stimulus)

        return np.sqrt(pca.explained_variance_)[-1]


class CellNeugebauer:
    # cell neugebauer with one division
    def __init__(self, primaries_dict: Dict[Union[Tuple, str], Spectra], n=50):
        """
        primaries_dict is (key, value) pairs where the key is either a
        string of ternary digits or a tuple of ternary values.
        """
        self.subcubes: Dict[Tuple, Neugebauer] = defaultdict(Neugebauer)

        weights = []
        spectras = []
        for key, spectra in primaries_dict.items():
            if isinstance(key, str):
                key = tuple(map(lambda x: int(x), key))
            weights.append(key)
            spectras.append(spectra)

        self.num_inks = round(math.log(len(primaries_dict), 3))
        self.wavelengths = list(primaries_dict.values())[0].wavelengths

        self.n = n

        for indices in product([0, 1], repeat=self.num_inks):
            primaries = {}
            for weight, spectra in zip(weights, spectras):
                if all(weight[d] in (indices[d], indices[d] + 1) for d in range(self.num_inks)):
                    adjusted_weight = tuple(w - i for w, i in zip(weight, indices))
                    primaries[adjusted_weight] = spectra
            self.subcubes[indices] = Neugebauer(primaries, n=n)

    def mix(self, percentages: npt.NDArray) -> npt.NDArray:
        index = tuple((percentages > 0.5).astype(int))
        adjusted_percentages = 2 * (percentages - np.array(index) / 2)
        return self.subcubes[index].mix(adjusted_percentages)

    def observe(self, percentages: npt.NDArray, observer: npt.NDArray, illuminant: Optional[npt.NDArray] = None):
        if illuminant is None:
            illuminant = np.ones_like(self.wavelengths)
        return observe_spectra(self.mix(percentages), observer, illuminant)


class Neugebauer:
    def __init__(self, primaries_dict: Optional[Dict[Union[Tuple, str], Spectra]], n=50):
        """
        primaries_dict is (key, value) pairs where the key is either a
        string of binary digits or a tuple of binary values, and value is a Spectra.
        """
        weights = []
        spectras = []
        for key, spectra in primaries_dict.items():
            if isinstance(key, str):
                key = tuple(map(lambda x: int(x), key))
            weights.append(np.array(key))
            spectras.append(spectra.data)
        self.wavelengths = list(primaries_dict.values())[0].wavelengths
        self.weights_array = np.array(weights)
        self.spectras_array = np.power(np.array(spectras), 1.0 / n)
        self.num_inks = round(math.log(self.weights_array.shape[0], 2))

        if isinstance(n, np.ndarray):
            assert len(n) == len(self.wavelengths)

        self.n = n

    def batch_mix(self, percentages: npt.NDArray) -> npt.NDArray:
        w_p = ((self.weights_array * percentages[:, np.newaxis, :]) +
               (1 - self.weights_array) * (1 - percentages[:, np.newaxis, :]))
        w_p_prod = np.prod(w_p, axis=2, keepdims=True)

        result = np.power(np.matmul(w_p_prod.transpose(0, 2, 1), self.spectras_array), self.n).squeeze(axis=1)

        return result

    def mix(self, percentages: npt.NDArray) -> npt.NDArray:
        w_p = (self.weights_array * percentages) + (1 - self.weights_array) * (1 - percentages)
        w_p_prod = np.prod(w_p, axis=1, keepdims=True)

        result = np.power(np.matmul(w_p_prod.T, self.spectras_array), self.n)

        return result

    def observe(self, percentages: npt.NDArray, observer: npt.NDArray, illuminant: Optional[npt.NDArray] = None):
        if illuminant is None:
            illuminant = np.ones_like(self.wavelengths)
        if percentages.ndim > 1:
            return observe_spectras(self.batch_mix(percentages), observer, illuminant)
        return observe_spectra(self.mix(percentages), observer, illuminant)

    def get_pca_size(self, observe: npt.NDArray, illuminant: npt.NDArray):
        # built for very quick evaluation of gamut
        stepsize = 0.2
        values = np.arange(0, 1 + stepsize, stepsize)
        mesh = np.meshgrid(*([values] * self.num_inks))
        grid = np.stack(mesh, axis=-1).reshape(-1, self.num_inks)

        stimulus = self.observe(grid, observe, illuminant)
        pca = PCA(n_components=observe.shape[0])
        pca.fit(stimulus)
        return np.sqrt(pca.explained_variance_)[-1]


class InkGamut:
    def __init__(self, inks: Union[List[Spectra], Neugebauer, CellNeugebauer, Dict[Union[Tuple, str], Spectra]],
                 paper: Optional[Spectra] = None,
                 illuminant: Optional[Union[Spectra, npt.NDArray]] = None):
        if isinstance(inks, Dict):
            inks = Neugebauer(inks)
        if isinstance(inks, Neugebauer) or isinstance(inks, CellNeugebauer):
            self.wavelengths = inks.wavelengths
        else:
            self.wavelengths = inks[0].wavelengths

        self.illuminant = None
        if isinstance(illuminant, Spectra):
            self.illuminant = illuminant.interpolate_values(self.wavelengths).data
        elif isinstance(illuminant, np.ndarray):
            assert len(illuminant) == len(self.wavelengths)
            self.illuminant = illuminant
        else:
            self.illuminant = np.ones_like(self.wavelengths)

        if isinstance(inks, Neugebauer) or isinstance(inks, CellNeugebauer):
            self.neugebauer = inks
            return

        assert np.array_equal(self.wavelengths, paper.wavelengths), \
            "Must pass in paper spectra with consistent wavelengths"

        for ink in inks:
            assert np.array_equal(ink.wavelengths, self.wavelengths)

        self.neugebauer = load_neugebauer(inks, paper)  # KM interpolation


    def get_spectra(self, percentages: Union[List, npt.NDArray], clip=False):
        if not isinstance(percentages, np.ndarray):
            percentages = np.array(percentages)
        data = self.neugebauer.mix(percentages).T
        if clip:
            data = np.clip(data, 0, 1)
        return Spectra(data=data, wavelengths=self.wavelengths)


    def batch_generator(self, iterable, batch_size):
        """Utility function to generate batches from an iterable."""
        iterator = iter(iterable)
        for first in iterator:
            yield np.array(list(islice(chain([first], iterator), batch_size - 1)))

    def get_spectral_point_cloud(self,stepsize=0.1, grid: Optional[npt.NDArray] = None, verbose=True, batch_size=1e5):

        point_cloud = []
        _percentages = []

        if grid is None:
            values = np.arange(0, 1 + stepsize, stepsize)
            total_combinations = len(values) ** self.neugebauer.num_inks
            grid = product(values, repeat=self.neugebauer.num_inks)
        else:
            total_combinations = grid.shape[0]

        desc = "Generating point cloud"
        verbose_progress = (lambda x: tqdm(x, total=int(total_combinations / batch_size), desc=desc)) if verbose else (lambda x: x)

        if isinstance(self.neugebauer, CellNeugebauer):
            for index, neugebauer in self.neugebauer.subcubes.items():
                subgamut = InkGamut(neugebauer, illuminant=self.illuminant)
                pc, perc = subgamut.get_spectral_point_cloud(stepsize=stepsize * 2, grid=None, verbose=verbose, batch_size=batch_size)
                point_cloud.append(pc)
                _percentages.append(perc / 2 + np.array(index) / 2)

        else:
            for batch in verbose_progress(self.batch_generator(grid, int(batch_size))):
                valid_percentages = batch[np.all(batch <= 1, axis=1)]

                if valid_percentages.size == 0:
                    continue

                spectral_batch = self.neugebauer.batch_mix(valid_percentages)
                point_cloud.append(spectral_batch)
                _percentages.append(valid_percentages)

        # Concatenate the batched results
        point_cloud = np.concatenate(point_cloud, axis=0)
        _percentages = np.concatenate(_percentages, axis=0)

        return point_cloud, _percentages

    def get_point_cloud(self, observe: Union[Observer, npt.NDArray],
                        stepsize=0.1, grid: Optional[npt.NDArray] = None, verbose=True, batch_size=1e5):
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(wavelengths=self.wavelengths)

        point_cloud = []
        _percentages = []

        if grid is None:
            values = np.arange(0, 1 + stepsize, stepsize)
            total_combinations = len(values) ** self.neugebauer.num_inks
            grid = product(values, repeat=self.neugebauer.num_inks)
        else:
            total_combinations = grid.shape[0]

        desc = "Generating point cloud"
        verbose_progress = (lambda x: tqdm(x, total=int(total_combinations / batch_size), desc=desc)) if verbose else (lambda x: x)

        if isinstance(self.neugebauer, CellNeugebauer):
            
            for index, neugebauer in self.neugebauer.subcubes.items():
                subgamut = InkGamut(neugebauer, illuminant=self.illuminant)
                pc, perc = subgamut.get_point_cloud(observe, stepsize=stepsize * 2, grid=None, verbose=verbose, batch_size=batch_size)
                point_cloud.append(pc)
                _percentages.append(perc / 2 + np.array(index) / 2)

        else:
            for batch in verbose_progress(self.batch_generator(grid, int(batch_size))):
                valid_percentages = batch[np.all(batch <= 1, axis=1)]

                if valid_percentages.size == 0:
                    continue

                stimulus_batch = self.neugebauer.observe(valid_percentages, observe, self.illuminant)
                point_cloud.append(stimulus_batch)
                _percentages.append(valid_percentages)

        # Concatenate the batched results
        point_cloud = np.concatenate(point_cloud, axis=0)
        _percentages = np.concatenate(_percentages, axis=0)

        return point_cloud, _percentages


    def get_buckets(self, observe: Union[Observer, npt.NDArray],
                  axis=2, stepsize=0.1, verbose=True, save=False):
        point_cloud, percentages = self.get_point_cloud(observe, stepsize, verbose=verbose)
        if verbose: print("Point cloud generated.")

        if save:
            np.save(f"{save}_point_cloud{int(100 * stepsize)}", point_cloud)
            np.save(f"{save}_percentages{int(100 * stepsize)}", percentages)
            if verbose: print(f"Point cloud saved to {save}_point_cloud{int(100 * stepsize)}.")

        _percentages = []

        buckets = sort_buckets(bucket_points(point_cloud, axis=axis), axis=axis)
        for dst, (i, j) in buckets:
            _percentages.append((dst, (tuple(percentages[i]), tuple(percentages[j]))))

        _percentages.sort(reverse=True)
        return _percentages
    
    def get_buckets_in_hering(self, max_basis: MaxBasis,
                  axis=2, stepsize=0.1, verbose=True, save=False):
        maxbasis_observer = max_basis.get_max_basis_observer()
        point_cloud, percentages = self.get_point_cloud(maxbasis_observer, stepsize, verbose=verbose)
        if verbose: print("Point cloud generated.")

        if save:
            np.save(f"{save}_point_cloud{int(100 * stepsize)}", point_cloud)
            np.save(f"{save}_percentages{int(100 * stepsize)}", percentages)
            if verbose: print(f"Point cloud saved to {save}_point_cloud{int(100 * stepsize)}.")

        _percentages = []
        # HMatrix = getHeringMatrix(4)
        Tmat = max_basis.get_cone_to_maxbasis_transform()
        # maxbasis_pts = (HMatrix @ Tmat @ point_cloud.T).T
        Q_vec = [Tmat @np.array([0, 0, 1, 0]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0])] # Q direction
        # NOT DONE
        def gram_schmidt(vectors):
            basis = []
            for v in vectors:
                w = v - np.sum( np.dot(v,b)*b  for b in basis )
                if (w > 1e-10).any():  
                    basis.append(w/np.linalg.norm(w))
            return np.array(basis)
        A = gram_schmidt(Q_vec)
        new_point_cloud = (A @ point_cloud.T).T
        buckets = sort_buckets(bucket_points(new_point_cloud, axis=0), axis=0)
        for dst, (i, j) in buckets:
            _percentages.append((dst, (tuple(percentages[i]), tuple(percentages[j]))))

        _percentages.sort(reverse=True)
        return _percentages

    def get_pca_size(self, observe: Union[Observer, npt.NDArray], stepsize=0.1, verbose=False):
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(wavelengths=self.wavelengths)
        point_cloud, _ = self.get_point_cloud(observe, stepsize, verbose=verbose)
        pca = PCA(n_components=observe.shape[0])
        pca.fit(point_cloud)

        return np.sqrt(pca.explained_variance_)[-1]



    def get_width(self, observe: Union[Observer, npt.NDArray],
                  axis=2, stepsize=0.1, verbose=True, save=False, refined=0):
        percentages = self.get_buckets(observe, axis=axis, stepsize=stepsize, verbose=verbose, save=save, refined=refined)

        dst, (pi, pj) = percentages[0]

        if verbose:
            print(f"maximum distance is {dst} with percentages {pi} and {pj}")

        return dst


