import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
from scipy.ndimage import zoom
from app.utils import imwrite, bin_array_to_shape

with PIL.Image.open('src/real_world.jpg', 'r') as h:
    img = np.array(h, dtype=np.int64)


img = img[:, :, :3]
img = img[30:768, 43:814]

densities = np.ones(shape=(img.shape[0], img.shape[1])) * 0.5

densities[img[:, :, 0] > 200] = 0.75
densities[img[:, :, 2] > 200] = 0.0

square_cut = min(densities.shape)

densities = densities[:square_cut, :square_cut]

densities = bin_array_to_shape(densities, (200,200))

final = np.ones(shape=densities.shape) * 0.5

final[densities == 0.0] = 0.0
final[densities == 0.75] = 0.75

np.save('real_world.npy', final)

plt.imshow(final)
plt.show()