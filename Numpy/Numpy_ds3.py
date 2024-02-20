# similarity

from sklearn.datasets import load_digits
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()

digit_1 = digits.images[0]
digit_2 = digits.images[10]
digit_3 = digits.images[1]
digit_4 = digits.images[11]

r_digit_1 = digit_1.reshape(64, 1)
r_digit_2 = digit_2.reshape(64, 1)
r_digit_3 = digit_3.reshape(64, 1)
r_digit_4 = digit_4.reshape(64, 1)

plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(1, 8, height_ratios=[1], width_ratios=[9, 1, 9, 1, 9, 1, 9, 1])

for i in range(4):
    plt.subplot(gs[2*i])
    plt.imshow(eval('digit_' + str(i + 1)), aspect=1, interpolation='nearest', cmap=plt.cm.bone_r)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title('images {}'.format(i + 1))
    plt.subplot(gs[2 * i + 1])
    plt.imshow(eval('r_digit_' + str(i + 1)), aspect=0.25, interpolation='nearest', cmap=plt.cm.bone_r)

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title('vector {}'.format(i + 1))
plt.tight_layout()
plt.show()


print((r_digit_1.T @ r_digit_2)[0][0])
print(r_digit_1)