import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])


print(np.concatenate((a, b[1:]), axis=0))
