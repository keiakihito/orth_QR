import numpy as np
from scipy.linalg import qr

# Given matrix
A = np.array([
    [1.0, 5.0, 9.0],
    [2.0, 6.0, 10.0],
    [3.0, 7.0, 11.0],
    [4.0, 8.0, 12.0]
])

# Perform QR decomposition with column pivoting
Q, R, P = qr(A, pivoting=True)

# Permutation matrix P
P_matrix = np.eye(A.shape[1])[:, P]

# Determine the rank by examining the diagonal of R
tolerance = 1e-6
rank = np.sum(np.abs(np.diag(R)) > tolerance)

# Truncate Q to the first 'rank' columns
Q_truncated = Q[:, :rank]

# Verify Q_truncated.T * Q_truncated equals the identity matrix
identity_check = np.allclose(Q_truncated.T @ Q_truncated, np.eye(rank))

print("Matrix A with column pivoting applied:")
print(A @ P_matrix)

print("\nMatrix Q (orthonormal):")
print(Q)

print("\nMatrix R (upper triangular):")
print(R)

print("\nPermutation indices P:")
print(P)

print("\nDetermined rank:")
print(rank)

print("\nQ_truncated:")
print(Q_truncated)

print("\nQ_truncated.T * Q_truncated:")
print(Q_truncated.T @ Q_truncated)

print("\nIs Q_truncated.T * Q_truncated close to the identity matrix?")
print(identity_check)
