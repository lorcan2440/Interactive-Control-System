import numpy as np

eigs = np.array([-0.5, -1.0, -2.0, 0.0, 1.0, -0.00000000001 + 0.00000000001j, -0.000000001 - 1j])


print(np.sum(np.isclose(eigs, 0)))