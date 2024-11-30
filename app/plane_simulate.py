import numpy as np
import matplotlib.pyplot as plt
from utils import update, start_fire_grid, generate_species, generate_rain
from plane_behaviour.plane import *
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

grid = np.load('results/processed.npy')

alpha_responder = planes(location=(10, 0), radius=4, speed=4)       # Initiate our plane at (10, 0) that has speed 2 cells with an effective radius of 4.

M, count_down = start_fire_grid(grid, p=0.00001)                    # Generate a grid

# We provide 4 different densities with varying burn and growth rates.
species = generate_species(shape=M.shape)
burning_time = [2, 2, 2, 2]
growing_time = [200, 225, 250, 275]

rain = generate_rain(shape=M.shape)
rain = np.zeros(shape=M.shape)

M = M + species/4

# Fix any artifacting from adding varying densities to our grid.
M[M < 0] = -1

M[30, 30] = 3
count_down[30, 30] = 2

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
fig, ax = plt.subplots(figsize=(12, 12))
ax.axis('off')
img = ax.imshow(M, cmap=cmap, norm=norm)

responder_marker, = ax.plot(alpha_responder.y, alpha_responder.x, 'ws', markersize=8, label="Responder")

# Animation update function
def animate(frame):
    global M, count_down
    M, count_down = update(M=M,
                           count_down=count_down,
                           species=species,
                           rain=rain,
                           burning_time=burning_time,
                           growing_time=growing_time,
                           wind_dir=[0, 10])

    alpha_responder.move(M)
    alpha_responder.extinguish(M)

    img.set_data(M)
    responder_marker.set_data([[alpha_responder.y], [alpha_responder.x]]) # nested list prevents deprecation warning.

    return [img , responder_marker]

# Create animation
N = 40  # Number of frames
anim = FuncAnimation(fig, animate, frames=N, interval=N, blit=True)

plt.legend()
plt.show()