import numpy as np
import numpy.typing as npt
from typing import List, Dict

from spectra import Spectra
from observer import Observer, Cone
from inks import InkGamut


def move(percentage: npt.NDArray, target_delta_smql, tetrachromat: Observer, simulated_gamut: InkGamut, delta=0.2):
    original_smql = tetrachromat.observe(simulated_gamut.get_spectra(percentage))

    ranges = [np.arange(p - delta, p + delta + 0.01, 0.05) for p in percentage]
    mesh = np.meshgrid(*ranges)

    best_score = -2
    best_percentage = None

    for values in zip(*[m.flatten() for m in mesh]):
        values = np.array(values)
        if any(values < 0) or any(values > 1): continue
        sample_spectra = simulated_gamut.get_spectra(values)

        new_smql = tetrachromat.observe(sample_spectra)
        delta_smql = new_smql - original_smql
        score = np.dot(delta_smql, target_delta_smql) / (np.linalg.norm(delta_smql) * np.linalg.norm(target_delta_smql))

        if score > best_score:
            best_score = score
            best_percentage = values

    return best_percentage, best_score > 0.5


def get_candidate_metamers(spectras: List[Spectra], tetrachromat: Observer, threshold=0.02):
    candidates = []
    for i in range(len(spectras)):
        for j in range(i, len(spectras)):
            smql_i = tetrachromat.observe(spectras[i])
            smql_j = tetrachromat.observe(spectras[j])

            deltas = np.abs(smql_i - smql_j)
            max_sml_delta = max(deltas[0], deltas[1], deltas[3])

            if deltas[2] > max_sml_delta and max_sml_delta < threshold:
                candidates.append((i, j))

    return candidates


def iterate_samples(samples_dict: Dict[str, Spectra], simulated_gamut: InkGamut, tetrachromat: Observer, threshold=0.02):
    # samples dict is to be populated in the usual way as nix data, e.g.
    # ("10, 20, 100, 50" : Spectra) is a key-pair

    next_samples = []
    percentages = [
        np.array([int(x)/100 for x in p.split(" ")])
        for p in samples_dict.keys()
    ]
    spectras = list(samples_dict.values())

    candidate_indices = get_candidate_metamers(list(samples_dict.values()), tetrachromat, threshold=threshold)
    for k, (i, j) in enumerate(candidate_indices):
        print(f"iterating {k}/{len(candidate_indices) -1}")
        next_samples.extend([percentages[i], percentages[j]])
        # print("Before", percentages[i], percentages[j])

        smql_i = tetrachromat.observe(spectras[i])
        smql_j = tetrachromat.observe(spectras[j])

        target_delta_i = (smql_j - smql_i)
        target_delta_i[2] = -1/(100*100*target_delta_i[2])
        target_delta_i = target_delta_i / np.linalg.norm(target_delta_i)

        target_delta_j = -target_delta_i

        updated_percentage_i, f1 = move(percentages[i], target_delta_i, tetrachromat, simulated_gamut)
        updated_percentage_j, f2 = move(percentages[j], target_delta_j, tetrachromat, simulated_gamut)

        if f1:
            # print("target delta i", target_delta_i)

            next_samples.append(updated_percentage_i)
            # print("updated i:", updated_percentage_i)
        if f2:
            # print("target delta j", target_delta_j)
            next_samples.append(updated_percentage_j)
            # print("updated j:", updated_percentage_j)

    prev_percentages = {
        tuple(int(round(100 * x)) for x in p) for p in percentages
    }
    new_percentages = {
        tuple(int(round(100 * x)) for x in p) for p in next_samples
    }

    return new_percentages - prev_percentages

def examine_samples(samples_dict, tetrachromat: Observer, threshold=0.02):
    # tells us how good the samples are
    sample_views = []
    detailed_views = []
    percentages = [
        np.array([int(x)/100 for x in p.split(" ")])
        for p in samples_dict.keys()
    ]
    spectras = list(samples_dict.values())
    all_pairs = get_candidate_metamers(spectras, tetrachromat, threshold=threshold)

    for (i, j) in all_pairs:
        s_i = spectras[i]
        s_j = spectras[j]
        smql_i = tetrachromat.observe(s_i)
        smql_j = tetrachromat.observe(s_j)

        deltas = np.abs(smql_i - smql_j)
        max_sml_delta = max(deltas[0], deltas[1], deltas[3])

        sample_views.append(
            (
                deltas[2],
                max_sml_delta
            )
        )

        detailed_views.append(
            (
                deltas[2],
                max_sml_delta,
                (percentages[i], smql_i),
                (percentages[j], smql_j)
            )
        )

    return sorted(sample_views, reverse=True), sorted(detailed_views, reverse=True)

