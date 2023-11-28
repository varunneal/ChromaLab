from typing import List, Tuple, Dict, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from math import log2

from spectra import Spectra, Pigment
from observer import Observer
from inks import Neugebauer, ThNeugebauer, k_s_from_pigments


class ColorMatchingModel(nn.Module):
    def __init__(self, num_primaries: int):
        super(ColorMatchingModel, self).__init__()
        initial_weights = th.full((num_primaries,), 1.0 / num_primaries)
        self.weights = nn.Parameter(initial_weights)

    def forward(self):
        concentrations = F.softmax(self.weights, dim=0)
        return concentrations


class NeugabauerMatchingModel(nn.Module):
    def __init__(self, num_primaries: int):
        super(NeugabauerMatchingModel, self).__init__()
        initial_weights = th.full((num_primaries,), 1.0 / num_primaries)
        self.weights = nn.Parameter(initial_weights)

    def forward(self):
        percentages = th.clamp(self.weights, 0, 1)
        return percentages


class EmptyModel(nn.Module):
    def __init__(self, initial_weights: th.Tensor):
        super(EmptyModel, self).__init__()
        self.weights = nn.Parameter(initial_weights)

    def forward(self):
        return self.weights




class PolynomialRootsModel(nn.Module):
    # its no good
    def __init__(self, domain, degree):
        super(PolynomialRootsModel, self).__init__()

        self.domain = th.tensor(domain).to(th.float32)

        self.roots1 = nn.Parameter(th.rand(degree))
        self.roots2 = nn.Parameter(th.rand(degree))

        self.coeff1 = nn.Parameter(th.tensor([1.0]))
        self.coeff2 = nn.Parameter(th.tensor([1.0]))

        self.offset1 = nn.Parameter(th.tensor([1.0]))
        self.offset2 = nn.Parameter(th.tensor([0.5]))

    def polynomial_with_roots(self, x, roots, coeff, offset):
        prod = th.ones_like(x)
        for root in roots:
            prod = prod * (x - root)
        result = prod * coeff + offset
        return result

    def forward(self):
        spectra1 = self.polynomial_with_roots(self.domain, self.roots1, self.coeff1, self.offset1)
        spectra2 = self.polynomial_with_roots(self.domain, self.roots2, self.coeff2, self.offset2)

        return th.clip(th.cat((spectra1, spectra2), dim=0), 0, 1)

class PowerSeriesModel(nn.Module):
    def __init__(self, domain, degree):
        super(PowerSeriesModel, self).__init__()
        self.power_series = th.stack([th.tensor(domain) ** i for i in range(degree)]).to(th.float32)
        self.factorials = th.cumprod(th.cat((th.tensor([1.0]), th.arange(1, degree, dtype=th.float32))), 0)

        self.coeffs1 = nn.Parameter(th.rand(degree))
        self.coeffs2 = nn.Parameter(th.rand(degree))

    def forward(self):
        spectra1 = th.sigmoid(th.matmul(self.coeffs1 / self.factorials, self.power_series))
        spectra2 = th.sigmoid(th.matmul(self.coeffs2 / self.factorials, self.power_series))
        return th.cat((spectra1, spectra2), dim=0)


class ClampModel(nn.Module):
    def __init__(self, initial_weights: th.Tensor):
        super(ClampModel, self).__init__()
        self.weights = nn.Parameter(initial_weights)

    def forward(self):
        percentages = th.clamp(self.weights, 0, 1)
        return percentages

class SigmoidModel(nn.Module):
    def __init__(self, initial_weights: th.Tensor):
        super(SigmoidModel, self).__init__()
        self.weights = nn.Parameter(initial_weights)

    def forward(self):
        percentages = th.sigmoid(self.weights)
        return percentages

def observe_spectra_th(spectra: th.Tensor, observer: th.Tensor, whitepoint: th.Tensor):
    return th.divide(th.matmul(observer, spectra.T).squeeze(), whitepoint)


def full_spectra_dual_optimization_metric(spectra: th.Tensor, observer: th.Tensor,
                                          whitepoint: th.Tensor, axis: int):
    midpoint = spectra.shape[0] // 2
    spectra1, spectra2 = spectra[:midpoint], spectra[midpoint:]
    # xprint(spectra1)

    # dimensions might be wrong
    pred1 = th.divide(th.matmul(observer, spectra1).squeeze(), whitepoint)
    pred2 = th.divide(th.matmul(observer, spectra2).squeeze(), whitepoint)

    mask_close = th.ones_like(pred1, dtype=th.bool)
    mask_close[axis] = 0

    mse_loss = th.mean((pred1[mask_close] - pred2[mask_close]) ** 2)
    magnitude_penalty = (pred1[axis] - pred2[axis]) ** 2

    return th.sqrt(mse_loss.sum() / magnitude_penalty)


def polynomial_dual_optimization_metric(coefficients: th.Tensor, power_series: th.Tensor, observer: th.Tensor,
                                          whitepoint: th.Tensor, axis: int):
    # Scale coefficients by factorial
    factorials = th.cumprod(th.cat((th.tensor([1.0]), th.arange(1, coefficients.shape[0], dtype=th.float32))), 0)
    scaled_coefficients = coefficients / factorials

    midpoint = scaled_coefficients.shape[0] // 2
    coefficients1, coefficients2 = scaled_coefficients[:midpoint], scaled_coefficients[midpoint:]

    spectra1, spectra2 = th.matmul(coefficients1, power_series), th.matmul(coefficients2, power_series)

    pred1 = th.divide(th.matmul(observer, spectra1).squeeze(), whitepoint)
    pred2 = th.divide(th.matmul(observer, spectra2).squeeze(), whitepoint)

    mask_close = th.ones_like(pred1, dtype=th.bool)
    mask_close[axis] = 0

    mse_loss = th.mean((pred1[mask_close] - pred2[mask_close]) ** 2)
    magnitude_penalty = (pred1[axis] - pred2[axis]) ** 2

    return th.sqrt(mse_loss.sum() / magnitude_penalty)



def transition_optimization_metric(transition_indices: th.Tensor, observer: th.Tensor, whitepoint: th.tensor, axis: int):
    # doesnt work because indices arent differentiable operations
    num_wavelengths = observer.shape[1]

    transition1 = th.zeros(num_wavelengths)
    transition2 = th.zeros(num_wavelengths)

    print("transition 1", transition1.grad_fn)

    midpoint = transition_indices.shape[0] // 2
    transition_indices1 = transition_indices[:midpoint] * num_wavelengths
    transition_indices2 = transition_indices[midpoint:] * num_wavelengths

    for i in range(0, len(transition_indices1), 2):
        a, b = transition_indices1[i], transition_indices1[i + 1]
        if a < b:
            transition1 = transition1.clone().scatter_(0, th.arange(int(a), int(b)), 1)

        c, d = transition_indices2[i], transition_indices2[i + 1]

        if c < d:
            transition2 = transition2.clone().scatter_(0, th.arange(int(c), int(d)), 1)

    pred1 = observe_spectra_th(transition1, observer, whitepoint)
    pred2 = observe_spectra_th(transition2, observer, whitepoint)

    print("pred 1", pred1.grad_fn)

    mask_close = th.ones_like(pred1, dtype=th.bool)
    mask_close = mask_close.clone().scatter_(0, th.tensor([axis]), 0)

    mse_loss = th.mean((pred1[mask_close] - pred2[mask_close]) ** 2)
    magnitude_penalty = (pred1[axis] - pred2[axis]) ** 2

    return th.sqrt(mse_loss.sum() / magnitude_penalty)



def km_matching_metric(concentrations: th.Tensor, target_rgb: th.Tensor,
                       observer: th.Tensor, pigments: List[Pigment]) -> th.float:
    K_matrix, S_matrix = k_s_from_pigments(pigments)

    K_mix = th.matmul(K_matrix, concentrations)
    S_mix = th.matmul(S_matrix, concentrations)

    R_mix = 1 + (K_mix / S_mix) - th.sqrt(th.square(K_mix / S_mix) + (2 * K_mix / S_mix))
    # todo: given reflectance coefficients, can account for surface reflection.
    #  Okumura • 2005 is a database of such coefficients.

    final_pred = th.matmul(observer, R_mix)

    l2_loss = ((final_pred - target_rgb) ** 2).sum()

    return l2_loss


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))



def train():
    ## neugebauer train
    # need to intitialize primaries_dict

    pass


def trainKM():
    wavelengths = np.arange(400, 701)

    """
    RGB Basis (approximation)
    """

    blue_curve = gaussian(wavelengths, 1.0, 450, 15)
    green_curve = gaussian(wavelengths, 0.75, 550, 15)
    red_curve = gaussian(wavelengths, 0.75, 600, 15) + gaussian(wavelengths, 0.2, 450, 15)

    # rgb_observer could also be created using an sRGB chromaticity matrix
    rgb_observer = th.from_numpy(np.vstack([red_curve, green_curve, blue_curve])).float()
    # XYZ observers can also use chromaticity matrices
    # conal observers (e.g. trichromats) can be specified easily using cone absorption data

    """
    CMYK Basis (bad approximation of this data https://www.diva-portal.org/smash/get/diva2:227132/FULLTEXT01.pdf)
    """
    cyan_curve = gaussian(wavelengths, 0.75, 645, 15)
    magenta_curve = gaussian(wavelengths, 0.6, 550, 50) + gaussian(wavelengths, 0.6, 400, 50)
    yellow_curve = gaussian(wavelengths, 0.8, 425, 15)
    key_curve = 0.05 * np.ones_like(wavelengths)  # black turns out to not just be zeroes in practice, so idk

    cyan = Pigment(Spectra(wavelengths=wavelengths, data=cyan_curve))
    magenta = Pigment(Spectra(wavelengths=wavelengths, data=magenta_curve))
    yellow = Pigment(Spectra(wavelengths=wavelengths, data=yellow_curve))
    key = Pigment(Spectra(wavelengths=wavelengths, data=key_curve))

    pigments = [cyan, magenta, yellow, key]

    """
    Color Matching
    """
    model = ColorMatchingModel(len(pigments))
    target_rgb = th.tensor([0.0, 0.39, 0.39])  # cyan

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    num_epochs = 100

    for epoch in range(1, num_epochs + 1):
        weights = model()

        loss = km_matching_metric(weights, target_rgb, rgb_observer, pigments)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not epoch % 10:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    print("Learned Concentrations:", F.softmax(model.weights.data, dim=0).numpy())

def grid_train(neugabauer: ThNeugebauer, observer: Observer, NUM_INITS=100, NUM_EPOCHS=100):
    param_size = neugabauer.num_inks
    observer_matrix = th.from_numpy(observer.get_sensor_matrix(neugabauer.wavelengths)).to(th.float32)
    whitepoint = th.from_numpy(observer.get_whitepoint(neugabauer.wavelengths)).to(th.float32)

    best_params = None
    best_loss = float('inf')

    for s in range(1, NUM_INITS + 1):
        initials = th.rand(2 * param_size)
        # mask = (th.rand(2 * param_size) > 0.66).float()
        # sparse_initials = initials * mask
        instance = ClampModel(initials)

        optimizer = optim.SGD(instance.parameters(), lr=0.05, momentum=0.9)

        instance_loss = float('inf')
        instance_params = None
        this_init = True

        count = 0
        while this_init:
            for epoch in range(1, NUM_EPOCHS + 1):
                percentages = instance()
                loss = neugabauer.dual_optimization_metric(
                    percentages, observer_matrix, whitepoint, 2
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_params = instance().detach()
            curr_loss = neugabauer.dual_optimization_metric(curr_params, observer_matrix, whitepoint, 2).item()

            # not decreasing (anymore)
            if curr_loss >= 0.99 * instance_loss or np.isnan(curr_loss):
                break

            instance_loss = curr_loss
            instance_params = curr_params

            count += 1
            # if count == 10 or count == 100:
            #     print(count, instance_loss)

        if instance_loss < best_loss:
            best_loss = instance_loss
            best_params = instance_params

        if not (s % (NUM_INITS // 10)):
            print(f"After {s} initializations, best loss is {best_loss} with parameters {best_params.detach()}.")

    return best_params



if __name__ == '__main__':
    train()
