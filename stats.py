# coding: utf-8
import numpy
import scipy.stats as stats

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def print_basic_indicators(series):
    print 'Min:', numpy.amin(series)
    print 'Max', numpy.amax(series)
    print 'Mean:', numpy.mean(series)
    print 'Variance:', numpy.var(series)


def symmetric_kl(distrib_p, distrib_q):
    return numpy.sum([stats.entropy(distrib_p, distrib_q), stats.entropy(distrib_p, distrib_q)])
