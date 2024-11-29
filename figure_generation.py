import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from app.plane_behaviour.clustering import find_fire_clusters

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



data = np.load('output.npy')
plot_data = data[25]
norm_plot = plot_data[35:79, 29:73]
plot_data = plot_data[30:90, 30:90]

fig = plt.figure(figsize=(5, 5))
plt.imshow(norm_plot, cmap=cmap, norm=norm)
plt.axis('off')
plt.show()

clusters, my_clustered_grid = find_fire_clusters(plot_data)


fig = plt.figure(figsize=(5,5))
plt.axis('off')


colors = ["#000000", "#cd001a", "#ef6a00", "#f2cd00", "#79c300", "#1961ae","#61007d"]

# Define the bounds corresponding to the values -- bounds are not inclusive so we have to add a little bit, 0.1, to our actual values.
bounds = [-0.1, 0.9, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1]  

# Create the colormap and norm
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)



plt.imshow(my_clustered_grid, cmap=cmap, norm=norm)

for i in range(1, len(clusters) + 1):
    curr_cluster = clusters[i]
    curr_cell = curr_cluster['cells'][0]
    plt.scatter(curr_cell[1], curr_cell[0], label=f'Cluster {i}', color=colors[i], marker='s')

plt.legend()

plt.show()