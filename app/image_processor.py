import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
from utils import imwrite, bin_array_to_shape
import json

with open('settings.json', 'r') as f:
  settings = json.load(f)


with PIL.Image.open(settings['image_path'], 'r') as h:
    img = np.array(h, dtype=np.int64)

img = img[:, :, :3]
img[:, :, 2] = 0
img[:, :, 1] = 0
img[img[:, :, 0] > 210] = 0

# W_0 = np.array([[1,1,1],
#                 [1,1,1],
#                 [1,1,1]]
#               )

# img_linear = linear_filter(img, W_0, mode = "reflect")

img_linear = np.sum(img, axis=2)

if settings['quality'] == 'high':
   img_linear = bin_array_to_shape(img_linear, (300, 300))
elif settings['quality'] == 'medium':
   img_linear = bin_array_to_shape(img_linear, (200, 200))
elif settings['quality'] == 'low':
   img_linear = bin_array_to_shape(img_linear, (100, 100))
else:
  raise RuntimeError("Invalid Simulation Quality Setting. Please check settings.json")

img_linear[img_linear > 0] = 255

np.save('results/processed.npy', img_linear)
imwrite('results/processed.png', img_linear)

plt.figure(figsize=(6, 6))
plt.imshow(img_linear)
plt.title("Processed Image")
plt.figtext(0.05, 0.05, "CHECK FOR ARTIFACTS!")
plt.axis('off')
plt.show()