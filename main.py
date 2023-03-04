import numpy as np
from numpy import linalg

from mock_dataset import dataset

A = np.array([
    [1, x] for x in dataset.X
])

weights = linalg.inv(np.dot(A.transpose(), A)).dot(A.transpose()).dot(dataset.Y)

restored_Y = A.dot(weights)

rrv = dataset.Y - restored_Y

sse = rrv.dot(rrv.transpose())

print(sse)
