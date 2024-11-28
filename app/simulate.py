import numpy as np
import matplotlib.pyplot as plt
from utils import update, start_fire_grid, generate_species
from plane_behaviour.plane import *
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

o, p = 100, 100
grid = np.load('results/processed.npy')
grid = grid[:100, :100]

alpha_responder = planes(10, 0, speed=2)

M, count_down = start_fire_grid(grid, p=0.00001)
M[50,50] = 3
species = generate_species(shape=M.shape)
burning_time = [2, 3, 4, 5]
growing_time = [40, 45, 50, 55]

M = M + species/4
M[M < 0] = -1

M[50, 50] = 3
count_down[50, 50] = 2

#density = np.random.randint(1,6,size=(m,n))
m = np.size(M,0) 
n = np.size(M,1)
density = np.zeros([m, n]) + 1

# Define the colors for each value
colors = ["#38afcd",       # River (soft sky blue)
          
          "#051f20",       # Species 1 flammable (forest green)
          "#0b2b26",       # Species 2 flammable (olive green)
          "#163832",       # Species 3 flammable (dark olive green)
          "#235347",       # Species 4 flammable (sage green)
                 

          "#186118",       # Species 1 growing (light green)
          "#145314",       # Species 2 growing (pale green)
          "#114611",       # Species 3 growing (muted teal green)
          "#0e380e",       # Species 4 growing (very light teal)

          "#4e1003",       # Burnt (dark gray)
          "#FF4500",       # On fire (orange-red)
          "#D8BFD8"        # Extinguished (lavender)
          ]


# Define the bounds corresponding to the values
bounds = [-1.1, -0.1, 0.1, 0.251, 0.51, 0.751, 1.1, 1.251, 1.51, 1.751, 2.1, 3.1, 4.1]

# Create the colormap and norm
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)


# Setup plot
fig, ax = plt.subplots()
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

anim.save('results/fire_sim.gif', fps=N)
print('VIEW YOUR SIMULATION IN THE `results` FOLDER')

plt.legend()
plt.show()