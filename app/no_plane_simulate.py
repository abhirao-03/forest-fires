import numpy as np
import matplotlib.pyplot as plt
from utils import update, start_fire_grid, generate_species, generate_rain
from plane_behaviour.plane import *
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

grid = np.load('results/processed.npy')

M, count_down = start_fire_grid(grid, p=0.00000001)                    # Generate a grid

# We provide 4 different densities with varying burn and growth rates.
species = generate_species(shape=M.shape)
burning_time = [2, 3, 4, 5]
growing_time = [200, 225, 250, 275]

rain = generate_rain(shape=M.shape)
rain = np.zeros(shape=M.shape)

M = M + species/4

# Fix any artifacting from adding varying densities to our grid.
M[M < 0] = -1

M[158, 144] = 3
count_down[158, 144] = 2

# M[40, 80] = 3
# count_down[40, 80] = 2

# M[40, 85] = 3
# count_down[40, 85] = 2

# M[45, 90] = 3
# count_down[45, 90] = 2

# Define the colors for each value
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

# Setup plot
fig, ax = plt.subplots(figsize=(5, 5))
plt.tight_layout()
ax.axis('off')
img = ax.imshow(M, cmap=cmap, norm=norm)

# Animation update function
def animate(frame):
    global M, count_down
    M, count_down = update(M=M,
                           count_down=count_down,
                           species=species,
                           rain=rain,
                           burning_time=burning_time,
                           growing_time=growing_time,
                           wind_dir=[-10, 10])


    img.set_data(M)

    return [img]

# Create animation
N = 40  # Number of frames
anim = FuncAnimation(fig, animate, frames=N, interval=1, blit=True)

plt.show()