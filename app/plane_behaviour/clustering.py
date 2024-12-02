# This effective radius is what controls our clustering.
# Each cluster considers the neighbouring cells using the effective radius and groups them together.
# The more cells that are connected in this pattern, the higher the size.
effective_radius = [
                  (-2, -1), (-2, 0), (-2, 1),
        (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
        ( 0, -2), ( 0, -1),          ( 0, 1), ( 0 , 2),
        ( 1, -2), ( 1, -1), ( 1, 0), ( 1, 1), ( 1, 2),
                  (2, -1),  ( 2, 0), ( 2, 1)
    ]


def find_fire_clusters(in_grid):
    # grid preprocessing to only consider the burning cells.
    grid = in_grid.copy()
    grid[grid != 3] = int(0)
    grid[grid == 3] = int(1)
    
    rows, cols = len(grid), len(grid[0])

    visited = [[False] * cols for _ in range(rows)]         # generate an array of True/False values to track visited/non-visted cells.

    clusters = {}               # store clusters in a dictionary containing the size and the cells that are contained in the cluster.
    cluster_id = 1              # start labelling clusters with id 1

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited[i][j]:  # new unvisited fire cell found
                
                # initialize a new cluster
                cluster_size = 0
                cluster_cells = []
                stack = [(i, j)]  # start DFS from this cell
                
                # Perform DFS to find all connected cells in this cluster
                while stack:
                    x, y = stack.pop()                  # take the last element from the appended list and continue.

                    if visited[x][y]:
                        continue

                    visited[x][y] = True                # since we've visited this cell set it to True.
                    
                    # add this cell to our clusters dictionary and add 1 to the cluster_size.
                    cluster_size += 1
                    cluster_cells.append((x, y))
                    grid[x][y] = cluster_id             # mark cell with the current cluster ID
                    
                    # check all neighbors using the hexagonal neighbouring
                    for dx, dy in effective_radius:
                        nx, ny = x + dx, y + dy

                        # make sure we're within the bounds of the grid.
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 1:
                            stack.append((nx, ny))
                
                # store cluster details in the dictionary
                clusters[cluster_id] = {'size': cluster_size, 'cells': cluster_cells}
                cluster_id += 1

    return clusters, grid