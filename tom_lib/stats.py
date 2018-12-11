# coding: utf-8
import numpy
import numpy as np
import scipy.stats as stats

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def symmetric_kl(distrib_p, distrib_q):
    return numpy.sum([stats.entropy(distrib_p, distrib_q), stats.entropy(distrib_p, distrib_q)])


def myjaccard(r_i, r_j):
    return len(set.intersection(set(r_i), r_j)) / float(len(set.union(set(r_i), r_j)))


def average_jaccard(r_i, r_j):
    if not r_i or not r_j:
        raise Exception("Ranked lists should have at least one element.")
    if len(r_i) != len(r_j):
        raise Exception("Both ranked term list should have the same dimension.")
    jacc_sum = [myjaccard(r_i[:d + 1], r_j[:d + 1]) for d in range(len(r_i))]
    return sum(jacc_sum) / float(len(r_i))


def jaccard_similarity_matrix(s_x, s_y):
    k = len(s_x)
    m = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            m[i, j] = average_jaccard(s_x[i], s_y[j])
    return m


def agreement_score(s_x, s_y):
    if not s_x or not s_y:
        raise Exception("The sets of ranked lists should have at least one element.")
    if len(s_x) != len(s_y):
        raise Exception("Both ranked term list sets should have the same dimension.")
    k = len(s_x)
    m = jaccard_similarity_matrix(s_x, s_y)
    agree_sum = [m[i,].max() for i in range(k)]
    return sum(agree_sum) / k
