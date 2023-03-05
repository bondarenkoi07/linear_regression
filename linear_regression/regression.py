import numpy as np
import math

from numpy import linalg


def linear_regression(
    independent, dependent, polynom_power
) -> tuple[float, float, np.matrix]:
    """creating linear regression"""
    A = np.array(
        [
            [math.pow(x, power) for power in range(0, polynom_power + 1)]
            for x in independent
        ]
    )

    weights = linalg.inv(np.dot(A.transpose(), A)).dot(
        A.transpose().dot(dependent)
    )  # веса на основе эталонных значений

    restored_Y = A.dot(weights)  # восстановление зависимой переменной

    rrv = dependent - restored_Y  # вектор регрессионных остатков

    sse = rrv.dot(rrv.transpose())  # сумма квадратов регрессионных остатков

    mse = sse / len(dependent)  # усредненнная ошибка

    return sse, mse, restored_Y
