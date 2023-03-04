import math
import random

import numpy as np


def __analytic_func__(x):
    return math.log(x) + 3 * math.sin(x) + random.lognormvariate(0, 1)


X = np.arange(0.001, 600, 0.001)
Y = np.array([__analytic_func__(x) for x in X])
