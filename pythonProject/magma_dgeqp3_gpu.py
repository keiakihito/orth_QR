import numpy as np
from scipy.linalg import qr

# Given matrix

#Case 1
# A = np.array([
#     [1.0, 5.0, 9.0],
#     [2.0, 6.0, 10.0],
#     [3.0, 7.0, 11.0],
#     [4.0, 8.0, 12.0]
# ])

#case2
A = np.array([
    [1.1, 0.8, 3.0, 2.2, 0.2, 0.7],
    [2.2, 1.6, 4.1, 3.3, 0.3, 0.8],
    [3.3, 2.4, 5.2, 4.4, 0.4, 1.1],
    [4.4, 3.2, 6.3, 5.5, 0.5, 1.5],
    [5.5, 2.3, 0.7, 1.7, 0.6, 3.2]
])
A = np.transpose(A)

#case3
# A = np.array([
#     [1.1, 0.8, 3.0, 2.2, 1.4, 0.6, 7.7],
#     [2.2, 1.6, 4.1, 3.3, 2.5, 1.5, 5.6],
#     [3.3, 2.4, 5.2, 4.4, 3.6, 2.4, 9.6],
#     [4.4, 3.2, 6.3, 5.5, 4.7, 3.3, 8.8],
#     [5.5, 4.0, 7.4, 6.6, 5.8, 4.2, 7.0]
# ])
# A = np.transpose(A)


#case4
# A = np.array([
#     [1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 0.1, 0.2, 0.3],
#     [2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 0.5, 0.7, 0.2],
#     [3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 0.3, 0.4, 0.5],
#     [4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 1.1, 1.2, 1.3],
#     [5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 1.9, 1.5, 1.8],
#     [6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 2.2, 2.3, 2.5],
#     [7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 2.9, 3.1, 3.2]
# ])
# A = np.transpose(A)

# #case5
# A = np.array([
#         [1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 7.7, 6.4, 8.6, 8.8, 6.0],
#         [2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 6.2, 7.6, 8.8, 7.0, 6.4],
#         [3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 5.4, 6.5, 7.6, 8.2, 8.8],
#         [4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 8.8, 7.3, 9.7, 9.9, 7.1],
#         [5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.1, 8.5, 9.9, 8.1, 7.3],
#         [6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 6.3, 7.6, 8.9, 9.6, 10.3],
#         [7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 9.9, 8.2, 0.8, 9.0, 8.2],
#         [8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.1, 9.2, 8.2, 7.2, 8.7, 0.2, 0.5, 1.8, 8.0, 9.4, 0.3, 9.2, 8.2],
#         [9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1, 8.1, 9.8, 1.5, 1.4, 2.4, 7.2, 8.7, 0.2, 0.4, 1.8],
#         [1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.2, 9.0, 0.9, 2.8, 2.8, 3.0, 1.0, 9.1, 1.9, 2.2, 0.4],
#         [2.2, 1.9, 3.1, 3.3, 1.5, 1.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 9.9, 1.2, 2.2, 1.4, 0.7],
#         [3.3, 2.8, 4.2, 4.4, 2.6, 2.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 9.0, 0.9, 2.8, 2.8, 3.0],
#         [4.4, 3.7, 5.3, 5.5, 3.7, 3.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 2.1, 0.7, 2.0, 3.2, 1.5],
#         [5.5, 4.6, 6.4, 6.6, 4.8, 4.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 0.8, 2.1, 3.2, 2.3, 1.0],
#         [6.6, 5.5, 7.5, 7.7, 5.9, 5.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 0.9, 1.0, 1.1, 1.2, 1.3]
# ])
# A = np.transpose(A)

# Perform QR decomposition with column pivoting
Q, R, P = qr(A, pivoting=True)

# Permutation matrix P
P_matrix = np.eye(A.shape[1])[:, P]

# Reconstruct original matrix with column pivoting applied
A_permuted = A @ P_matrix

print("Original Matrix A:")
print(A)

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
