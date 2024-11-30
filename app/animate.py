import numpy as np
import matplotlib.pyplot as plt
from utils import update, start_fire_grid, generate_species, generate_rain
from plane_behaviour.plane import *

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

large_M = M.copy()

M[158, 144] = 3
count_down[158, 144] = 2

M = M[76:226, 76:226]
count_down = count_down[76:226, 76:226]

steps = 100
M_tracked = np.zeros(shape=(steps, large_M.shape[0], large_M.shape[1]))

for i in range(50):
    print(f"on step {i}")
    M, count_down = update(M=M,
                           count_down=count_down,
                           species=species,
                           rain=rain,
                           burning_time=burning_time,
                           growing_time=growing_time,
                           wind_dir=[-10, 10])
    
    large_M[76:226, 76:226] = M

    M_tracked[i, :, :] = large_M

np.save('M_tracked.npy', M_tracked)