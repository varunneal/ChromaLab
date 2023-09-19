import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import List, Tuple

import numpy as np

from spectra import Spectra, Pigment


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


wavelengths = np.arange(400, 701)


class ColorMatchingModel(nn.Module):
    def __init__(self, num_paints: int):
        super(ColorMatchingModel, self).__init__()
        initial_weights = th.full((num_paints,), 1.0 / num_paints)
        self.weights = nn.Parameter(initial_weights)

    def forward(self):
        concentrations = F.softmax(self.weights, dim=0)
        return concentrations


def k_s_from_pigments(pigments: List[Pigment]) -> Tuple[th.Tensor, th.Tensor]:
    k_list = []
    s_list = []

    for pigment in pigments:
        k, s = pigment.get_k_s()
        k_list.append(k)
        s_list.append(s)

    k_matrix = th.from_numpy(np.column_stack(k_list)).float()
    s_matrix = th.from_numpy(np.column_stack(s_list)).float()

    return k_matrix, s_matrix


def matching_metric(concentrations: th.Tensor, target_rgb: th.Tensor,
                    observer: th.Tensor, pigments: List[Pigment]) -> th.float:
    K_matrix, S_matrix = k_s_from_pigments(pigments)

    K_mix = th.matmul(K_matrix, concentrations)
    S_mix = th.matmul(S_matrix,  concentrations)

    R_mix = 1 + (K_mix / S_mix) - th.sqrt(th.square(K_mix / S_mix) + (2 * K_mix / S_mix))

    final_pred = th.matmul(observer, R_mix)

    l2_loss = ((final_pred - target_rgb) ** 2).sum()

    return l2_loss


def train():
    """
    RGB Basis (approximation)
    """

    blue_curve = gaussian(wavelengths, 1.0, 450, 15)
    green_curve = gaussian(wavelengths, 0.75, 550, 15)
    red_curve = gaussian(wavelengths, 0.75, 600, 15) + gaussian(wavelengths, 0.2, 450, 15)

    rgb_observer = th.from_numpy(np.vstack([red_curve, green_curve, blue_curve])).float()

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

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 100

    for epoch in range(1, num_epochs + 1):
        weights = model()

        loss = matching_metric(weights, target_rgb, rgb_observer, pigments)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not epoch % 10 and epoch:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    print("Learned Concentrations:", F.softmax(model.weights.data, dim=0).numpy())


train()
