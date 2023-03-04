""" Набор данных для одномерной регрессии"""

import math
import random

import numpy as np


def __analytic_func__(x):
    """ Функция, определяющая зависимость набора данных Y от X @returns dependent value"""
    return math.log(x) + 3 * math.sin(x) + random.lognormvariate(0, 1) % 10


X = np.arange(0.001, 600, 0.001)
Y = np.array([__analytic_func__(x) for x in X])
