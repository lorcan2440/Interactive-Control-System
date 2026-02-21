import numpy as np
from scipy.linalg import expm

x_0 = np.array([[1.0], [1.0]])
t_span = np.arange(0, 0.1 + 0.01, 0.01)  # shape (11,)
A = np.array([[-1, 2], 
              [-3, -4]])

Bu = np.array([[1.0], 
               [0.0]])

# we want to find the (2, 11) array exp_A_t_times_x_0, where the i-th column is expm(A * t_span[i]) @ x_0
# create the (11, 2, 2) array of A * t_span
A_t_span = np.outer(t_span, A.flatten()).reshape(t_span.shape[0], A.shape[0], A.shape[1])  # shape (11, 2, 2)
exp_A_t_span = expm(A_t_span)  # shape (11, 2, 2)
print(exp_A_t_span)
exp_A_t_times_x_0 = exp_A_t_span @ x_0  # shape (11, 2, 1)
#print(exp_A_t_times_x_0[:, :, 0].T)  # shape (11, 2)

print('----')

# compute the (11, 2, 2) array of exp(A t) - I where I is the (dims x dims) identity matrix
exp_minus_I = exp_A_t_span - np.eye(A.shape[0])  # shape (11, 2, 2)
print(exp_minus_I)

print('-----')

# compute A^-1 @ (exp(A t) - I) @ Bu for each t in t_span, resulting in a (11, 2, 1) array
A_inv = np.linalg.inv(A)  # shape (2, 2)
A_inv_exp_minus_I_Bu = A_inv @ exp_minus_I @ Bu  # shape (11, 2, 1)
print(A_inv_exp_minus_I_Bu)
