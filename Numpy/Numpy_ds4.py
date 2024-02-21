import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces


faces = fetch_olivetti_faces()

f, ax = plt.subplots(1, 3)

ax[0].imshow(faces.images[6], cmap=plt.cm.bone)
ax[0].grid(False)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('imgae1')

ax[1].imshow(faces.images[10], cmap=plt.cm.bone)
ax[1].grid(False)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('image2')

n_faces = faces.images[6] * 0.6 + faces.images[10] * 0.4
print(n_faces.shape)
ax[2].imshow(n_faces, cmap=plt.cm.bone)
ax[2].grid(False)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title('new face(image1 + image2)')

plt.show()




