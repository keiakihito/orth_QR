import numpy as np
from scipy.linalg import qr

# Given matrix

#Case 1
A = np.array([
    [1.0, 5.0, 9.0],
    [2.0, 6.0, 10.0],
    [3.0, 7.0, 11.0],
    [4.0, 8.0, 12.0]
])

#case2
#case3
#case4
#case5

# Perform QR decomposition with column pivoting
Q, R, P = qr(A, pivoting=True)

# Permutation matrix P
P_matrix = np.eye(A.shape[1])[:, P]

# Reconstruct original matrix with column pivoting applied
A_permuted = A @ P_matrix

print("Matrix A with column pivoting applied:")
print(A_permuted)

print("\nMatrix Q (orthonormal):")
print(Q)

print("\nMatrix R (upper triangular):")
print(R)

print("\nPermutation indices P:")
print(P)

# Manually extracting the lower triangular part and tau values
# Note: In scipy.linalg.qr, tau is not explicitly returned, so this is a conceptual extraction
tau = np.diag(Q)  # This is a simplification, actual tau extraction might be different

# Printing the values for understanding
print("\nExtracted Householder vectors (lower triangular part of Q) and tau values:")
print("Householder vectors (lower triangular part of Q):")
print(np.tril(Q, -1))
print("Tau values:")
print(tau)

# Now we have:
# - Upper triangular matrix R from QR decomposition
# - Householder vectors implicitly in the lower triangular part of Q
# - Tau values (here assumed simplistically as diagonal of Q for demonstration)
