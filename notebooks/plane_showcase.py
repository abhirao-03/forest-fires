import numpy as np
import matplotlib.pyplot as plt
import app.plane as plane

grid = np.load('planes/testing/M_tracked.npy')  # bring in our tracked grid of fire.

alpha_responder = plane.planes(40, 60)          # define a plane at (40, 60).

curr_grid = grid[10].copy()                     # create a copy of our grid.
clustered_grid = curr_grid.copy()               # this grid will be used to cluster fires together.

# NOTE: lines 14-16 have been combined into a larger .move() function.
locs = alpha_responder.get_min_distance_cell_from_cluster(clustered_grid)   # score each fire and return an ordered list of cells based on priority.
target = locs[0]                                                            # focus on the first priority fire cluster.
alpha_responder.helper_move(target)                                         # move towards it.
alpha_responder.extinguish(curr_grid)                                       # extinguish it if its burning.

# Showcasing extinguishment of fire.
plt.imshow(grid[10])
plt.scatter(60, 40, label='plane_init')
plt.scatter(alpha_responder.y, alpha_responder.x, label='plane_next')
plt.legend()
plt.title('non_extinguished')
plt.show()

plt.imshow(curr_grid)
plt.scatter(60, 40, label='plane_init')
plt.scatter(alpha_responder.y, alpha_responder.x, label='plane_next')
plt.legend()
plt.title('extinguished')
plt.show()