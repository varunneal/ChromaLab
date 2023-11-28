from itertools import combinations
from typing import List, Union, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from typing import Tuple

import torch as th

from math import log2


from spectra import Spectra, Pigment
from tqdm import tqdm
from itertools import product
from observer import Observer
from collections import defaultdict


def bucket_points(points, axis=2):
    buckets = defaultdict(list)

    for idx, point in enumerate(points):
        key = tuple([round(c, 2) for i, c in enumerate(point) if i != axis])
        value = (tuple(point), idx)
        buckets[key].append(value)

    return buckets


def max_dist(buckets, axis=2):
    max_dist = 0
    best_indices = None, None
    for metamers in buckets.values():
        if len(metamers) <= 1:
            continue

        axis_values = [metamer[0][axis] for metamer in metamers]

        min_val = min(axis_values)
        max_val = max(axis_values)

        distance = max_val - min_val

        if distance > max_dist:
            max_dist = distance

            min_index = axis_values.index(min_val)
            max_index = axis_values.index(max_val)

            best_indices = (metamers[min_index][1], metamers[max_index][1])

    return max_dist, best_indices


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


def km_mix(pigments, concentrations=None):
    K_matrix, S_matrix = k_s_from_pigments(pigments)
    wavelengths = pigments[0].wavelengths()

    if not concentrations:
        concentrations = np.array([1 / len(pigments)] * len(pigments))

    K_mix = K_matrix @ concentrations
    S_mix = S_matrix @ concentrations / (len(pigments) ** 2)  # varuns secret correction term

    k = np.column_stack((wavelengths, K_mix))
    s = np.column_stack((wavelengths, S_mix))

    return Pigment(k=k, s=s)

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
            mixed_ink = km_mix(inks_to_mix)
            primaries_dict[binary_str] = mixed_ink
    return Neugebauer(primaries_dict)


def observe_spectra_th(spectra: th.Tensor, observer: th.Tensor, whitepoint: th.Tensor):
    return th.divide(th.matmul(observer, spectra.T).squeeze(), whitepoint)


def observe_spectra(spectra, observer, whitepoint):
    return np.divide(np.matmul(observer, spectra.T).squeeze(), whitepoint)



class Neugebauer:
    def __init__(self, primaries_dict: Dict[Union[Tuple, str], Spectra], n=50):
        """
        primaries_dict is (key, value) pairs where the key is either a
        string of binary digits or a tuple of binary values.
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
        self.num_inks = int(log2(self.weights_array.shape[0]))
        self.n = n

    def mix(self, percentages: npt.NDArray) -> npt.NDArray:
        w_p = (self.weights_array * percentages) + (1 - self.weights_array) * (1 - percentages)
        w_p_prod = np.prod(w_p, axis=1, keepdims=True)

        result = np.power(np.matmul(w_p_prod.T, self.spectras_array), self.n)

        return result

    def observe(self, percentages: npt.NDArray, observer: npt.NDArray, whitepoint: npt.NDArray):
        return observe_spectra(self.mix(percentages), observer, whitepoint)





class ThNeugebauer:
    def __init__(self, primaries_dict: Dict[Union[Tuple, str], Spectra], n=50, device="cpu"):
        weights = []
        spectras = []
        for key, spectra in primaries_dict.items():
            if isinstance(key, str):
                key = tuple(map(lambda x: int(x), key))
            weights.append(th.tensor(key, dtype=th.float32, device=device))
            spectras.append(th.from_numpy(spectra.data()).to(device, dtype=th.float32))
        self.wavelengths = list(primaries_dict.values())[0].wavelengths()
        self.weights_tensor = th.stack(weights).squeeze(1)
        self.spectras_tensor = th.pow(th.stack(spectras).squeeze(1), 1.0 / n)
        self.num_inks = int(log2(self.weights_tensor.shape[0]))
        self.n = n

    def mix(self, percentages: th.Tensor) -> th.Tensor:
        w_p = (self.weights_tensor * percentages) + (1 - self.weights_tensor) * (1 - percentages)
        w_p_prod = th.prod(w_p, dim=1, keepdim=True)

        result = th.pow(th.matmul(w_p_prod.T, self.spectras_tensor), self.n)

        return result

    def observe(self, percentages, observer, whitepoint):
        # Check if the input is a NumPy array and convert it to a PyTorch tensor if so
        if isinstance(percentages, np.ndarray):
            percentages = th.tensor(percentages, dtype=th.float32)
        if isinstance(observer, np.ndarray):
            observer = th.tensor(observer, dtype=th.float32)
        if isinstance(whitepoint, np.ndarray):
            whitepoint = th.tensor(whitepoint, dtype=th.float32)

        return observe_spectra_th(self.mix(percentages), observer, whitepoint)

    def metric(self, percentages: th.Tensor, target_stimulus: th.Tensor, observer: th.Tensor,
               whitepoint: th.Tensor) -> th.float:
        prediction = self.observe(percentages, observer, whitepoint)
        return ((prediction - target_stimulus) ** 2).sum()

    def dual_optimization_metric(self, percentages: th.Tensor, observer: th.Tensor,
                                 whitepoint: th.Tensor, axis: int):
        epsilon = 1e-8

        midpoint = percentages.shape[0] // 2
        percentages1, percentages2 = percentages[:midpoint], percentages[midpoint:]

        pred1 = self.observe(percentages1, observer, whitepoint)
        pred2 = self.observe(percentages2, observer, whitepoint)

        mask_close = th.ones_like(pred1, dtype=th.bool)
        mask_close[axis] = 0

        mse_loss = th.mean((pred1[mask_close] - pred2[mask_close]) ** 2)
        magnitude_penalty = (pred1[axis] - pred2[axis]) ** 2

        return th.sqrt(mse_loss.sum() / magnitude_penalty)

class InkGamut:
    # todo: passing in neugebauer primaries
    def __init__(self, inks: Union[List[Pigment], Neugebauer], paper: Spectra):
        self.wavelengths = inks[0].wavelengths()
        assert np.array_equal(self.wavelengths, paper.wavelengths())

        for ink in inks:
            assert np.array_equal(ink.wavelengths(), self.wavelengths)

        if isinstance(inks, Neugebauer):
            self.neugebauer = inks
        else:
            self.neugebauer = load_neugebauer(inks, paper)  # KM interpolation

    def get_spectra(self, percentages: Union[List, npt.NDArray]):
        if not isinstance(percentages, np.ndarray):
            percentages = np.array(percentages)
        return Spectra(data=self.neugebauer.mix(percentages).T, wavelengths=self.wavelengths)

    def get_point_cloud(self, observe: Union[Observer, npt.NDArray],
                        whitepoint: Optional[npt.NDArray] = None, stepsize=0.1):
        if isinstance(observe, Observer):
            observe = observe.get_sensor_matrix(wavelengths=self.wavelengths)
        if whitepoint is None:
            whitepoint = np.matmul(observe, np.ones(observe.shape[1]))
        point_cloud = []
        _percentages = []
        values = np.arange(0, 1 + stepsize, stepsize)
        # Wrap the iterable with tqdm for a progress bar
        total_combinations = (len(values) ** self.neugebauer.num_inks)
        for percentages in tqdm(product(values, repeat=self.neugebauer.num_inks), total=total_combinations,
                                desc='Generating Point Cloud'):
            stimulus = self.neugebauer.observe(np.array(percentages), observe, whitepoint)
            point_cloud.append(stimulus)
            _percentages.append(percentages)
        return np.array(point_cloud), np.array(_percentages)

    def get_width(self, observe: Union[Observer, npt.NDArray], whitepoint: Optional[npt.NDArray] = None,
                  axis=2, stepsize=0.1, verbose=True, save=False):
        point_cloud, percentages = self.get_point_cloud(observe, whitepoint, stepsize)
        if verbose: print("Point cloud generated.")

        if save:
            np.save(f"{save}_point_cloud{int(100 * stepsize)}", point_cloud)
            np.save(f"{save}_percentages{int(100 * stepsize)}", percentages)
            if verbose: print(f"Point cloud saved to {save}_point_cloud{int(100 * stepsize)}.")

        buckets = bucket_points(point_cloud, axis=axis)

        dst, (i, j) = max_dist(buckets, axis=axis)

        if verbose: print(f"maximum distance is {dst} with percentages {percentages[i]} and {percentages[j]}")
        return dst