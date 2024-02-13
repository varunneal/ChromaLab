from itertools import combinations
from typing import List, Union, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from typing import Tuple, Iterable, List

import math

from spectra import Spectra, Pigment
from tqdm import tqdm
from itertools import product

from observer import Observer
from collections import defaultdict


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



# def bucket_points(points, axis=2):
#     # overlapping buckets
#     buckets = defaultdict(list)
#
#     for idx, point in enumerate(points):
#         ranges = [(int(1000 * round(p, 3)) - 5, int(1000 * round(p, 3))) for i, p in enumerate(point) if i != axis]
#         keys = product(*ranges)
#
#         for key in keys:
#             value = (tuple(point), idx)
#             buckets[tuple(key)].append(value)
#
#     return buckets

def bucket_points(points, axis=2):
    # disjointed buckets
    buckets = defaultdict(list)

    for idx, point in enumerate(points):
        key = tuple(int(100*round(p,2))for i, p in enumerate(point) if i != axis)
        value = (tuple(point), idx)
        buckets[key].append(value)
        # ranges = [(int(1000 * round(p, 3)) - 5, int(1000 * round(p, 3))) for i, p in enumerate(point) if i != axis]
        # keys = product(*ranges)
        #
        # for key in keys:
        #     value = (tuple(point), idx)
        #     buckets[tuple(key)].append(value)

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

def k_s_from_pigments(pigments):
    k_list = []
    s_list = []

    for pigment in pigments:
        if not isinstance(pigment, Pigment):
            pigment = Pigment(reflectance=pigment)
        k, s = pigment.get_k_s()
        k_list.append(k)
        s_list.append(s)

    k_matrix = np.column_stack(k_list)
    s_matrix = np.column_stack(s_list)

    return k_matrix, s_matrix


def km_mix(pigments, concentrations=None, coefficients=None):
    K_matrix, S_matrix = k_s_from_pigments(pigments)
    wavelengths = pigments[0].wavelengths()

    if not concentrations:
        concentrations = np.array([1 / len(pigments)] * len(pigments))

    K_mix = K_matrix @ concentrations
    S_mix = S_matrix @ concentrations

    return Pigment(k=K_mix, s=S_mix, wavelengths=wavelengths)


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


def observe_spectra(reflectance, observer, illuminant):
    return np.divide(np.matmul(observer, (reflectance * illuminant).T).squeeze(), np.matmul(observer, illuminant.T))


def find_best_n(primaries_dict, percentages: npt.NDArray, actual: Spectra):
    # Find best n array for a particular sample
    wavelengths = actual.wavelengths()

    best_n = np.ones_like(wavelengths)
    errors = np.ones_like(wavelengths)

    candidates = np.logspace(-1, 1, num=10, base=10)
    # print(candidates)
    # inefficient but idc
    for n in candidates:
        neug_n = Neugebauer(primaries_dict, n=n)
        result = neug_n.mix(percentages).reshape(-1)
        for i, (a, b) in enumerate(zip(result, actual.data())):
            error = (b-a) ** 2

            if error < errors[i]:
                errors[i] = error
                best_n[i] = n
    return best_n


def find_best_ns(primaries_dict, samples_dict):
    # samples dict will need to be (Tuple, Spectra) pairs
    best_ns = []

    # wavelengths = np.zeros_like(list(primaries_dict.values())[0].wavelengths())
    for percentage, actual in samples_dict.items():
        best_n = find_best_n(primaries_dict, np.array(percentage), actual)
        best_ns.append(best_n)

    return best_ns


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
        self.wavelengths = list(primaries_dict.values())[0].wavelengths()

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
    def __init__(self, primaries_dict: Dict[Union[Tuple, str], Spectra], n=50):
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
            spectras.append(spectra.data())
        self.wavelengths = list(primaries_dict.values())[0].wavelengths()
        self.weights_array = np.array(weights)
        self.spectras_array = np.power(np.array(spectras), 1.0 / n)
        self.num_inks = int(math.log(self.weights_array.shape[0], 2))

        if isinstance(n, np.ndarray):
            assert len(n) == len(self.wavelengths)

        self.n = n

    def mix(self, percentages: npt.NDArray) -> npt.NDArray:
        w_p = (self.weights_array * percentages) + (1 - self.weights_array) * (1 - percentages)
        w_p_prod = np.prod(w_p, axis=1, keepdims=True)

        result = np.power(np.matmul(w_p_prod.T, self.spectras_array), self.n)

        return result

    def observe(self, percentages: npt.NDArray, observer: npt.NDArray, illuminant: Optional[npt.NDArray] = None):
        if illuminant is None:
            illuminant = np.ones_like(self.wavelengths)
        return observe_spectra(self.mix(percentages), observer, illuminant)


class InkGamut:
    def __init__(self, inks: Union[List[Spectra], Neugebauer, CellNeugebauer], paper: Optional[Spectra] = None,
                 illuminant: Optional[Union[Spectra, npt.NDArray]] = None):
        if isinstance(inks, Neugebauer) or isinstance(inks, CellNeugebauer):
            self.wavelengths = inks.wavelengths
        else:
            self.wavelengths = inks[0].wavelengths()

        self.illuminant = None
        if isinstance(illuminant, Spectra):
            # assert np.array_equal(self.wavelengths, whitepoint.wavelengths())
            self.illuminant = illuminant.interpolate_values(self.wavelengths).data()
        elif isinstance(illuminant, np.ndarray):
            assert len(illuminant) == len(self.wavelengths)
            self.illuminant = illuminant
        else:
            self.illuminant = np.ones_like(self.wavelengths)

        if isinstance(inks, Neugebauer) or isinstance(inks, CellNeugebauer):
            self.neugebauer = inks
            return

        assert np.array_equal(self.wavelengths, paper.wavelengths())

        for ink in inks:
            assert np.array_equal(ink.wavelengths(), self.wavelengths)

        self.neugebauer = load_neugebauer(inks, paper)  # KM interpolation

    def get_spectra(self, percentages: Union[List, npt.NDArray], clip=False):
        if not isinstance(percentages, np.ndarray):
            percentages = np.array(percentages)
        data = self.neugebauer.mix(percentages).T
        if clip:
            data = np.clip(data, 0, 1)
        return Spectra(data=data, wavelengths=self.wavelengths)

    def get_refined_point_cloud(self, observer: Union[Observer, npt.NDArray], p: npt.NDArray, q: npt.NDArray,
                                stepsize=0.1, refined_stepsize=0.02):
        p0 = np.maximum(0, p - stepsize)
        q0 = np.maximum(0, q - stepsize)

        print(f"exploring {p0} to {np.minimum(p + stepsize, 1)} and {q0} to {np.minimum(q + stepsize, 1)}")

        fine_values = np.array(list(product(np.arange(0, 2 * stepsize + refined_stepsize, refined_stepsize),
                                            repeat=self.neugebauer.num_inks)))

        # handle in get_point_cloud instead
        # p_grid = np.minimum(p0 + np.array(list(fine_values)), 1)
        # q_grid = np.minimum(q0 + np.array(list(fine_values)), 1)

        p_point_cloud, p_percentages = self.get_point_cloud(observer,  grid=p_grid)
        q_point_cloud, q_percentages = self.get_point_cloud(observer, grid=q_grid)

        point_cloud = np.concatenate([p_point_cloud, q_point_cloud], axis=0)
        percentages = np.concatenate([p_percentages, q_percentages], axis=0)

        return point_cloud, percentages

    def get_point_cloud(self, observe: Union[Observer, npt.NDArray],
                        stepsize=0.1, grid: Optional[npt.NDArray] = None, verbose=True):
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(wavelengths=self.wavelengths)

        point_cloud = []
        _percentages = []

        if grid is None:
            values = np.arange(0, 1 + stepsize, stepsize)  # todo: this is wrong
            total_combinations = len(values) ** self.neugebauer.num_inks
            grid = product(values, repeat=self.neugebauer.num_inks)
        else:
            total_combinations = grid.shape[0]

        desc = "Generating point cloud"
        verbose_progress = (lambda x: tqdm(x, total=total_combinations, desc=desc)) if verbose else (lambda x: x)

        for percentages in verbose_progress(grid):
            arr = np.array(percentages)
            if np.any(arr > 1):
                continue

            stimulus = self.neugebauer.observe(arr, observe, self.illuminant)
            point_cloud.append(stimulus)
            _percentages.append(percentages)

        return np.array(point_cloud), np.array(_percentages)

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

        for t in range(refined):
            _, (pi, pj) = _percentages[t]
            refined_pc, refined_perc = self.get_refined_point_cloud(observe, np.array(pi), np.array(pj),  stepsize=2*stepsize)

            buckets = sort_buckets(bucket_points(refined_pc, axis=axis), axis=axis)
            for dst, (i, j) in buckets:
                _percentages.append((dst, (tuple(refined_perc[i]), tuple(refined_perc[j]))))

        _percentages.sort(reverse=True)
        return _percentages

    def get_width(self, observe: Union[Observer, npt.NDArray],
                  axis=2, stepsize=0.1, verbose=True, save=False, refined=False):
        percentages = self.get_buckets(observe, axis=axis, stepsize=stepsize, verbose=verbose, save=save, refined=refined)

        dst, (pi, pj) = percentages[0]

        if verbose:
            print(f"maximum distance is {dst} with percentages {pi} and {pj}")

        return dst
