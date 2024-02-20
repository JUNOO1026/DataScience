import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits, load_iris

iris = load_iris()
print(iris.data[0, :])
print(iris.data[1, :])

X = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]])

print(X)


digits = load_digits()

samples = [0, 10, 20, 30, 1, 11, 21, 31]
d = []

for i in range(len(samples)):
    d.append(digits.images[samples[i]])

plt.figure(figsize=(8, 2))

for i in range(len(samples)):
    plt.subplot(1, 8, i + 1)
    plt.imshow(d[i], interpolation='nearest', cmap=plt.cm.bone_r)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title('image {}'.format(i+1))

plt.suptitle('숫자 0과 1 이미지')
plt.tight_layout()
plt.show()

v = []

for i in range(len(samples)):
    v.append(d[i].reshape(64, 1))

plt.figure(figsize=(8, 4))
for i in range(len(samples)):
    plt.subplot(1, 8, i + 1)
    plt.imshow(v[i], interpolation='nearest', cmap=plt.cm.bone_r)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title('vector {}'.format(i+1))
plt.suptitle('vector image')
plt.tight_layout(w_pad=8)
plt.show()