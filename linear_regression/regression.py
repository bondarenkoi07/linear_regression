# pylint: disable=invalid-name
"""package providing linear regression build"""
import math
import numpy as np


def linear_regression(
    independent, dependent, polynom_power
) -> tuple[float, float, np.matrix, np.matrix]:
    """creating linear regression"""
    A = np.array(
        [
            [math.pow(x, power) for power in range(0, polynom_power + 1)]
            for x in independent
        ]
    )  # Vandermonde matrix

    weights = np.linalg.inv(np.dot(A.transpose(), A)).dot(
        A.transpose().dot(dependent)
    )  # веса на основе эталонных значений

    restored = A.dot(weights)  # восстановление зависимой переменной

    rrv = dependent - restored  # вектор регрессионных остатков

    sse = rrv.dot(rrv.transpose())  # сумма квадратов регрессионных остатков

    mse = sse / len(dependent)  # усредненная ошибка

    return sse, mse, restored, weights
