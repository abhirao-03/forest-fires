import numpy as np
import matplotlib.pyplot as plt
from utils import cmap, norm, update, start_fire_grid, generate_densities, generate_rain, start_fire_at
from plane_behaviour.plane import *
from matplotlib.animation import FuncAnimation
import json

with open('settings.json', 'r') as f:
  settings = json.load(f)

# Pull in user set fire_probability
p = settings['fire_probability']

# Pull in user set wind_direction
if settings['wind_direction'] == 'S':
    wind_dir = [0, -10]
elif settings['wind_direction'] == 'N':
    wind_dir = [0, 10]
elif settings['wind_direction'] == 'E':
    wind_dir = [-10, 0]
elif settings['wind_direction'] == 'W':
    wind_dir = [10, 0]
elif settings['wind_direction'] == 'NW':
    wind_dir = [10, 10]
elif settings['wind_direction'] == 'SW':
    wind_dir = [10, -10]
elif settings['wind_direction'] == 'NE':
    wind_dir = [-10, 10]
elif settings['wind_direction'] == 'SE':
    wind_dir = [-10, -10]

# If the user specified a personal image then use image_processing to convert it into usable grid.
if settings['use_personal_image'] == True:
    from image_processor import *
    grid = np.load('results/processed.npy')                     # It takes the quality from the set JSON.

elif settings['quality'] == "low":
    grid = np.zeros(shape=(100, 100))

elif settings['quality'] == "medium":
    grid = np.zeros(shape=(200, 200))

elif settings['quality'] == "high":
    grid = np.zeros(shape=(300, 300))

M, count_down = start_fire_grid(grid, p=p)                    # Generate a grid

# If the user specified varying densities, then create varying densities using Perlin noise.
if settings['has_varying_densities'] == True:
    # We provide 4 different densities with varying burn and growth rates.
    species = generate_densities(shape=M.shape, seed=settings['density_seed'])
    # species = np.load('real_world.npy') * 4                       # uncomment for real world
else:
    species = np.zeros(shape=M.shape)

# If the user specified rain, then create rain using Perlin noise.
if settings['has_rain'] == True:
    rain = generate_rain(shape=M.shape)
else:
    rain = np.zeros(shape=M.shape)

# Set the burn and growing time to be the user set list.
burning_time = settings['burn_time']
growing_time = settings['grow_time']

M = M + species/4

# Fix any artifacting from adding varying densities to our grid.
M[M < 0] = -1

# -----------------------------------------------------------------------------------------------------------------
#  Now the model is ready. Here the user can specify a fire location using `start_fire_at()`
# -----------------------------------------------------------------------------------------------------------------


M, count_down = start_fire_at(M, count_down, (30, 66))             # use this to start a fire.


# Uncomment below for real world fire start.

# M[100, 100] = 3
# count_down[100, 100] = 3

# M[80, 85:87] = 3
# count_down[80, 85:87] = 2

# M[81, 86:90] = 3
# count_down[81, 86:90] = 2

# M[82, 91] = 3
# count_down[82, 91] = 2

# M[82, 92] = 3
# count_down[82, 92] = 2

# M[83, 98] = 3
# count_down[83, 98] = 2

# M[84, 99] = 3
# count_down[84, 99] = 2

# M[84, 98] = 3
# count_down[84, 98] = 2

plt.figure(figsize=(6, 6))
plt.imshow(M, cmap=cmap, norm=norm)
plt.title('Initial Grid')
plt.figtext(0.05, 0.05, "CHECK FOR ISSUES -- e.g. fire started over water")
plt.show()

# Setup plot
fig, ax = plt.subplots(figsize=(12, 12))
ax.axis('off')
img = ax.imshow(M.T, cmap=cmap, norm=norm)

if settings['has_plane'] == True:
    alpha_responder = planes(location=(10,0), radius=settings["plane_radius"], speed=settings["plane_speed"])
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
                           wind_dir=wind_dir)

    img.set_data(M)

    if settings['has_plane'] == True:
        alpha_responder.move(M)
        alpha_responder.extinguish(M)
        responder_marker.set_data([[alpha_responder.y], [alpha_responder.x]])   # nested list prevents deprecation warning.

        return [img, responder_marker]
    
    else:
        return [img]

N = settings['time_steps']
anim = FuncAnimation(fig, animate, frames=N, interval=10, blit=True)
plt.tight_layout()
if settings['has_plane'] == True: plt.legend()
anim.save('results/fire_sim.gif')

plt.show()