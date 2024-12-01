import numpy as np
import matplotlib.pyplot as plt
from utils import update, start_fire_grid, generate_densities, generate_rain, cmap, norm
from plane_behaviour.plane import *
from matplotlib.animation import FuncAnimation
import json

#from image_processor import *

with open('settings.json', 'r') as f:
  settings = json.load(f)

grid = np.zeros(shape=(200,200))
M, count_down = start_fire_grid(grid, p=0.00001)                    # Generate a grid

if settings['varying_densities'] == True:
    # We provide 4 different densities with varying burn and growth rates.
    species = generate_densities(shape=M.shape)
    species = np.load('real_world.npy') * 4
    
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


M[80, 85:87] = 3
count_down[80, 85:87] = 2

M[81, 86:90] = 3
count_down[81, 86:90] = 2

M[82, 91] = 3
count_down[82, 91] = 2

M[82, 92] = 3
count_down[82, 92] = 2

M[83, 98] = 3
count_down[83, 98] = 2

M[84, 99] = 3
count_down[84, 99] = 2

M[84, 98] = 3
count_down[84, 98] = 2

plt.imshow(M)
plt.show()

# Setup plot
fig, ax = plt.subplots(figsize=(12, 12))
ax.axis('off')
img = ax.imshow(M.T, cmap=cmap, norm=norm)

if settings["plane"] == True:
    alpha_responder = planes(location=(10,0), speed=20)
    responder_marker, = ax.plot(alpha_responder.y, alpha_responder.x, 'ws', markersize=8, label="Responder")

# Animation update function
def animate(frame):
    plt.title(f"time step {frame}")
    global M, count_down
    M, count_down = update(M=M,
                           count_down=count_down,
                           species=species,
                           rain=rain,
                           burning_time=burning_time,
                           growing_time=growing_time,
                           wind_dir=[-10, 10])

    img.set_data(M)

    if settings["plane"] == True:
        alpha_responder.move(M)
        alpha_responder.extinguish(M)
        responder_marker.set_data([[alpha_responder.y], [alpha_responder.x]]) # nested list prevents deprecation warning.

        return [img, responder_marker]
    
    else:
        return [img]

# Create animation
N = 200  # Number of frames
anim = FuncAnimation(fig, animate, frames=N, interval=10, blit=True)
plt.tight_layout()
if settings['plane'] == True: plt.legend()
anim.save('AHHHH.gif')


plt.show()