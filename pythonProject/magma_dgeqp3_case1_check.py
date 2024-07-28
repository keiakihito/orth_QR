import numpy as np
from scipy.linalg import qr

# Given matrix
A = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
])

# Perform QR decomposition with column pivoting
Q, R, P = qr(A.T, pivoting=True)

# Permutation matrix P
P_matrix = np.eye(A.shape[0])[:, P]

# Reconstruct original matrix with column pivoting applied
A_permuted = A.T @ P_matrix

print("Matrix A with column pivoting applied:")
print(A_permuted)

print("\nR matrix:")
print(R)

print("\nTau values:")
tau = np.diag(R)[:-1]
print(tau)
