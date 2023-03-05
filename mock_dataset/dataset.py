""" Набор данных для одномерной регрессии"""

import math

import numpy as np


def __analytic_func__(arg):
    """Функция, определяющая зависимость набора данных Y от X @returns dependent value"""
    return math.log(arg) + 3 * math.sin(arg)


X = np.arange(0.001, 600, 0.001)
Y = np.array([__analytic_func__(x) for x in X])

X_CONTROL = np.arange(600.001, 1200, 0.001)
Y_CONTROL = np.array([__analytic_func__(x) for x in X_CONTROL])
