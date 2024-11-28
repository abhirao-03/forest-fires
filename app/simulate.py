import numpy as np
import matplotlib.pyplot as plt
from utils import update, start_fire_grid, generate_species
from plane_behaviour.plane import *
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

grid = np.load('results/processed.npy')

alpha_responder = planes(location=(10, 0), radius=4, speed=2)       # Initiate our plane at (10, 0) that has speed 2 cells with an effective radius of 4.

M, count_down = start_fire_grid(grid, p=0.00001)                    # Generate a grid

# We provide 4 different densities with varying burn and growth rates.
species = generate_species(shape=M.shape)
burning_time = [5, 4, 3, 2]
growing_time = [40, 45, 50, 55]

M = M + species/4

# Fix any artifacting from adding species to our grid.
M[M < 0] = -1
M[50, 50] = 3
count_down[50, 50] = 2

# Define the colors for each value
colors = ["#38afcd",       # River
          
          "#051f20",       # Species 1 flammable
          "#0b2b26",       # Species 2 flammable
          "#163832",       # Species 3 flammable
          "#235347",       # Species 4 flammable

          "#186118",       # Species 1 growing
          "#145314",       # Species 2 growing
          "#114611",       # Species 3 growing
          "#0e380e",       # Species 4 growing

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
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis('off')
ax.set_title("Fire Simulation")
img = ax.imshow(M, cmap=cmap, norm=norm)

responder_marker, = ax.plot(alpha_responder.y, alpha_responder.x, 'ws', markersize=8, label="Responder")

# Animation update function
def animate(frame):
    global M, count_down
    M, count_down = update(M, count_down, species, burning_time, growing_time, wind_dir=[0, 0])

    alpha_responder.move(M)
    alpha_responder.extinguish(M)

    img.set_data(M)
    responder_marker.set_data([[alpha_responder.y], [alpha_responder.x]]) # nested list prevents deprecation warning.

    return [img, responder_marker]

# Create animation
N = 100  # Number of frames
anim = FuncAnimation(fig, animate, frames=N, interval=10, blit=True)

anim.save('results/fire_sim.gif', fps=200, dpi=200)
print('VIEW YOUR SIMULATION IN THE `results` FOLDER')

plt.legend()
plt.show()