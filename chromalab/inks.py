import heapq
from itertools import combinations, chain
from typing import List, Union, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from itertools import product, islice, combinations


from typing import Tuple, Iterable, List
from tqdm import tqdm
from itertools import product

from collections import defaultdict
import math

from .spectra import Spectra
from .observer import Observer

from sklearn.decomposition import PCA
from scipy.special import comb


class Pigment(Spectra):
    # TODO: reformat with kwargs
    def __init__(self, array: Optional[Union[Spectra, npt.NDArray]] = None,
                 k: Optional[npt.NDArray] = None,
                 s: Optional[npt.NDArray] = None,
                 wavelengths: Optional[npt.NDArray] = None):
        """
        Either pass in @param reflectance or pass in
        @param k and @param s.

        k and s are stored as spectra rather than NDArrays.
        """
        if array is not None:
            # compute k & s from reflectance
            if isinstance(array, Spectra):
                super().__init__(array.array())
            else:
                super().__init__(array)
            _k, _s = self.compute_k_s()
            self.k, self.s = _k, _s

        elif k is not None and s is not None and wavelengths is not None:
            # compute reflectance from k & s
            if not k.shape == s.shape or not k.shape == wavelengths.shape:
                raise ValueError("Coefficients k and s and wavelengths must be same shape.")

            r = 1 + (k / s) - np.sqrt(np.square(k / s) + (2 * k / s))
            super().__init__(wavelengths=wavelengths, data=r)
        else:
            raise ValueError("Must either specify reflectance or k and s coefficients and wavelengths.")

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


def bucket_points(points, axis=2):
    # disjointed buckets
    buckets = defaultdict(list)

    for idx, point in enumerate(points):
        key = tuple(int(100*round(p,2))for i, p in enumerate(point) if i != axis)
        value = (tuple(point), idx)
        buckets[key].append(value)

    return buckets

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


def load_neugebauer(inks, paper):
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
    return Neugebauer(primaries_dict)


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

    best_n = np.ones_like(wavelengths)
    errors = np.ones_like(wavelengths)

    candidates = np.logspace(-1, 1, num=10, base=10)
    # print(candidates)
    # inefficient but idc
    for n in candidates:
        neug_n = Neugebauer(primaries_dict, n=n)
        result = neug_n.mix(percentages).reshape(-1)
        for i, (a, b) in enumerate(zip(result, actual.data)):
            error = (b-a) ** 2

            if error < errors[i]:
                errors[i] = error
                best_n[i] = n
    return best_n


def find_best_ns(primaries_dict, samples_dict):
    # samples dict will need to be (Tuple, Spectra) pairs
    best_ns = []

    for percentage, actual in samples_dict.items():
        best_n = find_best_n(primaries_dict, np.array(percentage), actual)
        best_ns.append(best_n)

    return best_ns


class InkLibrary:
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

        # Create a single tqdm progress bar
        with tqdm(total=total_iterations, desc="loading km cache") as pbar:
            for i in range(2, k + 1):
                concentrations = np.ones(i) / i
                for subset in combinations(range(self.K), i):
                    inks_to_mix = self.spectras[list(subset)]
                    ks_batch = []
                    for ink in inks_to_mix:
                        ks_batch.append(np.stack(k_s_from_data(ink)))
                    ks_batch = np.stack(ks_batch, axis=2)
                    ks_mix = ks_batch @ concentrations
                    data = data_from_k_s(ks_mix[0], ks_mix[1])
                    km_cache[subset] = data.astype(np.float16)
                    pbar.update(1)

        weights_array = []
        for i in range(k+1):
            for subset in combinations(range(k), i):
                binary = [0] * k
                for index in subset:
                    binary[index] = 1
                weights_array.append(binary)
        weights_array = np.array(weights_array)

        top_scores = []

        # Find best inkset
        for inkset_idx in tqdm(combinations(range(self.K), k), total=comb(self.K, k), desc="finding best inkset"):
            names = [self.names[i] for i in inkset_idx]

            primaries_array = [self.paper]
            primaries_array.extend([self.spectras[idx] for idx in inkset_idx])
            primaries_array.extend(
                [km_cache[subset] for i in range(2, k + 1) for subset in combinations(inkset_idx, i)])

            primaries_array = np.array(primaries_array)
            neug = FastNeugebauer(weights_array, primaries_array, k)
            score = neug.get_pca_size(observe, illuminant)

            if len(top_scores) < top:
                heapq.heappush(top_scores, (score, names))
            else:
                if score > top_scores[0][0]:
                    heapq.heapreplace(top_scores, (score, names))

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

    def get_pca_size(self, observe: npt.NDArray, illuminant: npt.NDArray):
        # built for very quick evaluation of gamut
        stepsize = 0.2
        values = np.arange(0, 1 + stepsize, stepsize)
        mesh = np.meshgrid(*([values] * self.num_inks))
        grid = np.stack(mesh, axis=-1).reshape(-1, self.num_inks)

        stimulus = self.batch_observe(grid, observe, illuminant)
        pca = PCA(n_components=observe.shape[0])
        pca.fit(stimulus)
        return np.sqrt(pca.explained_variance_)[-1]


class CellNeugebauer:
    # cell neugebauer with one division
    def __init__(self, primaries_dict: Dict[Union[Tuple, str], Spectra], illuminant: Optional[Spectra] = None, n=50):
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

        self.num_inks = int(math.log(len(primaries_dict), 3))
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
        self.num_inks = int(math.log(self.weights_array.shape[0], 2))

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
    def __init__(self, inks: Union[List[Spectra], Neugebauer, CellNeugebauer], paper: Optional[Spectra] = None,
                 illuminant: Optional[Union[Spectra, npt.NDArray]] = None):
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

    # def get_refined_point_cloud(self, observer: Union[Observer, npt.NDArray], p: npt.NDArray, q: npt.NDArray,
    #                             stepsize=0.1, refined_stepsize=0.02):
    #     p0 = np.maximum(0, p - stepsize)
    #     q0 = np.maximum(0, q - stepsize)
    #
    #     print(f"exploring {p0} to {np.minimum(p + stepsize, 1)} and {q0} to {np.minimum(q + stepsize, 1)}")
    #
    #     fine_values = np.array(list(product(np.arange(0, 2 * stepsize + refined_stepsize, refined_stepsize),
    #                                         repeat=self.neugebauer.num_inks)))
    #
    #     # handle in get_point_cloud instead
    #     # p_grid = np.minimum(p0 + np.array(list(fine_values)), 1)
    #     # q_grid = np.minimum(q0 + np.array(list(fine_values)), 1)
    #
    #     p_point_cloud, p_percentages = self.get_point_cloud(observer,  grid=p_grid)
    #     q_point_cloud, q_percentages = self.get_point_cloud(observer, grid=q_grid)
    #
    #     point_cloud = np.concatenate([p_point_cloud, q_point_cloud], axis=0)
    #     percentages = np.concatenate([p_percentages, q_percentages], axis=0)
    #
    #     return point_cloud, percentages




    def batch_generator(self, iterable, batch_size):
        """Utility function to generate batches from an iterable."""
        iterator = iter(iterable)
        for first in iterator:
            yield np.array(list(islice(chain([first], iterator), batch_size - 1)))

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
                  axis=2, stepsize=0.1, verbose=True, save=False, refined=0):
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

        # for t in range(refined):
        #     _, (pi, pj) = _percentages[t]
        #     refined_pc, refined_perc = self.get_refined_point_cloud(observe, np.array(pi), np.array(pj),  stepsize=2*stepsize)
        #
        #     buckets = sort_buckets(bucket_points(refined_pc, axis=axis), axis=axis)
        #     for dst, (i, j) in buckets:
        #         _percentages.append((dst, (tuple(refined_perc[i]), tuple(refined_perc[j]))))

        _percentages.sort(reverse=True)
        return _percentages

    def get_width(self, observe: Union[Observer, npt.NDArray],
                  axis=2, stepsize=0.1, verbose=True, save=False, refined=0):
        percentages = self.get_buckets(observe, axis=axis, stepsize=stepsize, verbose=verbose, save=save, refined=refined)

        dst, (pi, pj) = percentages[0]

        if verbose:
            print(f"maximum distance is {dst} with percentages {pi} and {pj}")

        return dst


