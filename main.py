"""find best polynom power for linear regression"""
import math

import matplotlib.pyplot as plt
import numpy as np

from linear_regression.regression import linear_regression
from mock_dataset import dataset

res = []

for p in range(1, 15):
    res.append(linear_regression(dataset.X, dataset.Y, p))

min_mse = min(res, key=lambda x: (x[0], x[1]))

index = res.index(min_mse)

A = np.array(
    [[math.pow(x, power) for power in range(0, index + 2)] for x in dataset.X_CONTROL]
)  # Vandermonde matrix

plt.ylim(top=15)
plt.plot(dataset.X, dataset.Y, label="original func value", color="g")
plt.plot(dataset.X_CONTROL, dataset.Y_CONTROL, label="control func value", color="y")
plt.plot(
    dataset.X, res[index][2], label="linear regression on original values", color="b"
)
plt.plot(
    dataset.X_CONTROL,
    A.dot(res[index][3]),
    label="linear regression on control values",
    color="r",
)
plt.ylabel("dependent variable")
plt.xlabel("independent variable")

plt.legend()

plt.show()

print(f"mse={res[index][1]} sse={res[index][0]} polynom power={index+1}")
