import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

img_rgb = misc.face()
print(img_rgb.shape)

plt.subplot(221)
plt.imshow(img_rgb, cmap=plt.cm.gray)
plt.axis('off')
plt.title('RGB color image')

plt.subplot(222)
plt.imshow(img_rgb[:, :, 0], cmap=plt.cm.gray)
plt.axis('off')
plt.title('RED color image')

plt.subplot(223)
plt.imshow(img_rgb[:, :, 1], cmap=plt.cm.gray)
plt.axis('off')
plt.title('GREEN color image')

plt.subplot(224)
plt.imshow(img_rgb[:, :, 2], cmap=plt.cm.gray)
plt.axis('off')
plt.title('BLUE color image')


plt.show()

