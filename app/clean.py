import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
from scipy.ndimage import zoom
from utils import imwrite, bin_array_to_shape


data = np.load('app/all_densities.npy')

copy_1 = data.copy()
copy_1[copy_1 == 0.0] = 0.6
copy_1[copy_1 == 0.5] = 0.0
copy_1[copy_1 == 0.6] = 0.5

copy_1 = bin_array_to_shape(copy_1, (300,300))

np.save("real_world.npy", copy_1)