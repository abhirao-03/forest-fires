import numpy as np
import matplotlib.pyplot as plt
from utils import update, start_fire_grid, generate_species
from plane import *
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

o, p = 100, 100
grid = np.load('imaging/processed.npy')

alpha_responder = planes(10, 0)

M, count_down = start_fire_grid(grid, p=0.00001)
M[50,50] = 3
species = generate_species(shape=M.shape)
burning_time = [2, 3, 4, 5]
growing_time = [50, 60, 70, 80]

M = M + species
M[M < 0] = -1

M[50, 50] = 3
count_down[50, 50] = 2

#density = np.random.randint(1,6,size=(m,n))
m = np.size(M,0) 
n = np.size(M,1)
density = np.zeros([m, n])+1

cmap = ListedColormap([ 'blue',
                       'darkgreen',
                       'seagreen',
                        'green',
                       'forestgreen',
                       'darkseagreen', 'grey', 'red', 'thistle'])
boundaries = [-1.1,
              -0.9,
               0.1,
               0.251,
               0.51,
               0.751,
               1.1,
               2.1,
               3.1,
               4.1]  # Boundaries to separate the color intervals

norm = BoundaryNorm(boundaries, cmap.N)  # Use norm to map boundaries to cmap indices


# Setup plot
fig, ax = plt.subplots()
ax.set_title("Fire Simulation")
img = ax.imshow(M, cmap=cmap, norm=norm)

responder_marker, = ax.plot(alpha_responder.y, alpha_responder.x, 'bo', markersize=8, label="Responder")

# Animation update function
def animate(frame):
    global M, count_down
    M, count_down = update(M, count_down, species, burning_time, growing_time, wind_dir=[100, -10])

    alpha_responder.move(M)
    alpha_responder.extinguish(M)

    img.set_data(M)
    responder_marker.set_data(alpha_responder.y, alpha_responder.x)

    return [img, responder_marker]

# Create animation
N = 100  # Number of frames
anim = FuncAnimation(fig, animate, frames=N, interval=10, blit=True)

# Show the animation
plt.legend()
plt.show()