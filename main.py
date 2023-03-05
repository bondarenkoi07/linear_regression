"""find best polynom power for linear regression"""
import matplotlib.pyplot as plt

from linear_regression.regression import linear_regression
from mock_dataset import dataset

res = []

for p in range(1, 14):
    res.append(linear_regression(dataset.X, dataset.Y, p))

min_mse = min(res, key=lambda x: (x[0], x[1]))

index = res.index(min_mse)

plt.plot(dataset.X, dataset.Y, label="original func value", color="g")
plt.plot(dataset.X, res[index][2], label="linear regression", color="b")
plt.ylabel("dependent variable")
plt.xlabel("independent variable")

plt.legend()

plt.show()

print(f"mse={res[index][1]} sse={res[index][0]} polynom power={index}")
