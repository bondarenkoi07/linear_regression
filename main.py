import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

from mock_dataset import dataset

A = np.array([[1, x] for x in dataset.X])

weights = linalg.inv(np.dot(A.transpose(), A)).dot(
    A.transpose().dot(dataset.Y)
)  # веса на основе эталонных значений

restored_Y = A.dot(weights)  # восстановление зависимой переменной

rrv = dataset.Y - restored_Y  # вектор регрессионных остатков

sse = rrv.dot(rrv.transpose())  # сумма квадратов регрессионных остатков

mse = sse / len(dataset.Y)  # усредненнная ошибка

plt.plot(dataset.X, dataset.Y, label="original func value", color="g")
plt.plot(dataset.X, restored_Y, label="linear regression", color="b")
plt.ylabel("dependent variable")
plt.xlabel("independent variable")

plt.legend()

plt.show()

print(f"mse={mse} sse={sse}")
