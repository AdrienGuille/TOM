# coding: utf-8
import numpy

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def print_basic_indicators(series):
    print 'Min:', numpy.amin(series)
    print 'Max', numpy.amax(series)
    print 'Mean:', numpy.mean(series)
    print 'Variance:', numpy.var(series)
