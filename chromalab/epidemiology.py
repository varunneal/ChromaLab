import random
from collections import Counter

import numpy as np

L_SNPs = ["116tyr", "180ala", "230thr"]
M_SNPs = ["180ser", "230iso"]

M_peaks = {
    (True, True): 536,
    (True, False): 533,
    (False, True): 533,
    (False, False): 530
}

L_peaks = {
    (True, True, True): 547,
    (True, True, False): 552,
    (True, False, True): 553,
    (True, False, False): 556.5,
    (False, True, True): 551,
    (False, True, False): 555,
    (False, False, True): 556,
    (False, False, False): 559
}

def m_cone(pdf, bias_factor=0.5):

    m_genotype = tuple(np.random.rand(2) < pdf)
    return M_peaks[m_genotype]


def l_cone(pdf, bias_factor=10.0):


    l_genotype = tuple(np.random.rand(3) < pdf)
    return L_peaks[l_genotype]

def peaks(case):
    if case == "lm":
        m_pdf = [0.067, 0.0024]
        l_pdf = [0.015, 0.34, 0.025]
        return [m_cone(m_pdf), l_cone(l_pdf)]
    elif case == "single-cone-protanope":
        m_pdf = [0.46, 0.15]
        return [m_cone(m_pdf)]
    elif case == "protanonomal":
        m_pdf = [0.017, 0.25]
        return [m_cone(m_pdf), m_cone(m_pdf)]
    elif case == "single-cone-deutanopes":
        l_pdf = [0.071, 0.25, 0.01]
        return [l_cone(l_pdf)]
    elif case == "deutanomal":
        l_pdf = [0.01, 0.518, 0.027]
        return [l_cone(l_pdf), l_cone(l_pdf)]
    else:
        raise ValueError(f"improper case {case}")


def get_random_x(n):
    genotypes = ["lm",
                 "single-cone-protanope", "protanonomal",
                 "single-cone-deutanopes", "deutanomal"]
    weights = [0.92,
               0.0029, 0.0181,
               0.0086, 0.005]
    genotype = random.choices(genotypes, weights, k=n)
    return [peaks(g) for g in genotype]

def simulate_males(n):
    return get_random_x(n)

def simulate_females(n):
    return [x1 + x2 for x1, x2 in zip(get_random_x(n), get_random_x(n))]

def functional_dimension(peaks):
    # counts functional dimensional of peaks
    return len(set(peaks))

def functional_genotype(peaks):
    return tuple(sorted(set(peaks)))


def dimensional_analysis(n=10000):

    male_dimensions = Counter([functional_dimension(peaks) for peaks in simulate_males(n)])
    female_dimensions = Counter([functional_dimension(peaks) for peaks in simulate_females(n)])

    return male_dimensions, female_dimensions


def genotype_analysis(n=10000):
    male_genotypes = Counter([
        functional_genotype(peaks) for peaks in simulate_males(n)
    ])
    female_genotypes = Counter([
        functional_genotype(peaks) for peaks in simulate_females(n)
    ])
    return male_genotypes, female_genotypes


def main():
    n = 100000

    # lm_snps()
    male_dimensions, female_dimensions = dimensional_analysis(n)
    male_simulations, female_simulations = genotype_analysis(n)

    print(male_simulations)
    print(female_simulations)

    for peaks, count in female_simulations.most_common():
        print(peaks, "\t", round(100 * (count / n), 4))


if __name__ == '__main__':
    main()