import numpy as np
from sklearn.datasets import load_digits

x = load_digits().data

x_1 = x[0].reshape(64, 1)
x_9 = x[9].reshape(64, 1)

print(x_1.shape)
print(x_9.shape)

print((x @ x.T)[0][0])


A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])

print(A @ B)
print(B @ A)


C = np.array([[1, 2, 3]])
D = np.array([[4, 7], [5, 8], [6, 9]])

print(C @ D)
# print(D @ C)

E = np.array([[1, 2], [3, 4]])
F = np.array([[5, 6], [7, 8]])

print(E @ F)
print(F @ E)