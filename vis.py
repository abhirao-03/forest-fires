import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

M_tracked = np.load('M_tracked.npy')
M_tracked = M_tracked[:, 76:226, 76:226]

colors = ["#38afcd",       # River
          
          "#051f20",       # Species 1 flammable
          "#0b2b26",       # Species 2 flammable
          "#163832",       # Species 3 flammable
          "#235347",       # Species 4 flammable

          "#3e3636",       # Species 1 growing
          "#5d5555",       # Species 2 growing
          "#716868",       # Species 3 growing
          "#999090",       # Species 4 growing

          "#4e1003",       # Burnt
          "#FF4500",       # On fire
          "#D8BFD8"        # Extinguished
          ]

# Define the bounds corresponding to the values -- bounds are not inclusive so we have to add a little bit, 0.1, to our actual values.
bounds = [-1.1, -0.1, 0.1, 0.251, 0.51, 0.751, 1.1, 1.251, 1.51, 1.751, 2.1, 3.1, 4.1]  

# Create the colormap and norm
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)


fig = plt.figure(figsize=(12, 12))

plt.imshow(M_tracked[1], cmap=cmap, norm=norm)
plt.axis('off')
plt.tight_layout()
plt.show()
