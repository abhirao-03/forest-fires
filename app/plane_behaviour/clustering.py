effective_radius = [
                  (-2, -1), (-2, 0), (-2, 1),
        (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
        ( 0, -2), ( 0, -1),          ( 0, 1), ( 0 , 2),
        ( 1, -2), ( 1, -1), ( 1, 0), ( 1, 1), ( 1, 2),
                  (2, -1),  ( 2, 0), ( 2, 1)
    ]

def find_fire_clusters(in_grid):
    grid = in_grid.copy()
    grid[grid != 3] = int(0)
    grid[grid == 3] = int(1)
    

    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    clusters = {}
    cluster_id = 2

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited[i][j]:  # New unvisited fire cell found
                # Initialize a new cluster
                cluster_size = 0
                cluster_cells = []
                stack = [(i, j)]  # Start DFS from this cell
                
                # Perform DFS to find all connected cells in this cluster
                while stack:
                    x, y = stack.pop()

                    if visited[x][y]:
                        continue

                    visited[x][y] = True
                    cluster_size += 1
                    cluster_cells.append((x, y))
                    grid[x][y] = cluster_id  # Mark cell with the current cluster ID
                    
                    # Check all neighbors
                    for dx, dy in effective_radius:
                        nx, ny = x + dx, y + dy

                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 1:
                            stack.append((nx, ny))
                
                # Store cluster details
                clusters[cluster_id] = {'size': cluster_size, 'cells': cluster_cells}
                cluster_id += 1

    return clusters