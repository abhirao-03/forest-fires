import numpy as np
import matplotlib.pyplot as plt
from utils import update, start_fire_grid, generate_densities, generate_rain, cmap, norm
from plane_behaviour.plane import *
from matplotlib.animation import FuncAnimation
import json

from image_processor import *

with open('settings.json', 'r') as f:
  settings = json.load(f)

grid = np.load('results/processed.npy')
M, count_down = start_fire_grid(grid, p=0.00001)                    # Generate a grid

if settings['varying_densities'] == True:
    # We provide 4 different densities with varying burn and growth rates.
    species = generate_densities(shape=M.shape)
    burning_time = settings['burn_time']
    growing_time = settings['grow_time']
else:
    species = np.zeros(shape=M.shape)

if settings['rain'] == True:
    rain = generate_rain(shape=M.shape)
else:
    rain = np.zeros(shape=M.shape)


M = M + species/4

# Fix any artifacting from adding varying densities to our grid.
M[M < 0] = -1


M[110, 28] = 3
count_down[110, 28] = 2

# Setup plot
fig, ax = plt.subplots(figsize=(12, 12))
ax.axis('off')
img = ax.imshow(M, cmap=cmap, norm=norm)

if settings["plane"] == True:
    alpha_responder = planes(location=(10,0), speed=10)
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
                           wind_dir=[0, 0])

    img.set_data(M)

    if settings["plane"] == True:
        alpha_responder.move(M)
        alpha_responder.extinguish(M)
        responder_marker.set_data([[alpha_responder.y], [alpha_responder.x]]) # nested list prevents deprecation warning.

        return [img , responder_marker]
    
    else:
        return [img]

# Create animation
N = 40  # Number of frames
anim = FuncAnimation(fig, animate, frames=N, interval=10, blit=True)

if settings['plane'] == True: plt.legend()
plt.show()