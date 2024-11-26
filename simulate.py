import numpy as np
import matplotlib.pyplot as plt
from utils import update, start_fire_grid
from plane import *
from matplotlib.animation import FuncAnimation

grid = np.load('imaging/processed.npy')
grid = grid[:100, :100]


alpha_responder = planes(40, 60)

M, count_down = start_fire_grid(grid)
M[50, 50] = 3
count_down[50, 50] = 2

#density = np.random.randint(1,6,size=(m,n))
m = np.size(M,0) 
n = np.size(M,1)
density = np.zeros([m, n])+1

fig, ax = plt.subplots()
artists = []
N = 100


# Setup plot
fig, ax = plt.subplots()
ax.set_title("Fire Simulation")
img = ax.imshow(M, cmap='hot', interpolation='nearest')

responder_marker, = ax.plot(alpha_responder.y, alpha_responder.x, 'bo', markersize=8, label="Responder")

# Animation update function
def animate(frame):
    global M, count_down
    M, count_down = update(M, count_down, burning_time=2, wind_dir=[1, -1])

    alpha_responder.move(M)
    alpha_responder.extinguish(M)

    img.set_data(M)
    responder_marker.set_data(alpha_responder.y, alpha_responder.x)

    return [img, responder_marker]

# Create animation
N = 100  # Number of frames
anim = FuncAnimation(fig, animate, frames=N, interval=100, blit=True)

# Show the animation
plt.legend()
plt.show()