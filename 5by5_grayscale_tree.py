import numpy as np
import math
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
import random
import numpy as np
import matplotlib.pyplot as plt

def create_d_adj_list(picture, width, height):
    """
    Create an adjacency list with neighbors and their square difference in darkness.
    """
    d_adj_list = {}

    for row in range(height):
        for col in range(width):
            current_index = row * width + col
            current_value = picture[current_index]  # Darkness value of the current pixel
            
            neighbors = []
            total_weight = 0
            for drow in [-1, 0, 1]:
                for dcol in [-1, 0, 1]:
                    if drow == 0 and dcol == 0:
                        continue
                    
                    neighbor_row = row + drow
                    neighbor_col = col + dcol
                    
                    if 0 <= neighbor_row < height and 0 <= neighbor_col < width:
                        neighbor_index = neighbor_row * width + neighbor_col
                        neighbor_value = picture[neighbor_index]
                        relationship = round(math.exp( -9*(current_value - neighbor_value) ** 2), 4)
                        # relationship = -abs(current_value - neighbor_value)
                        total_weight += relationship
                        neighbors.append((neighbor_index, relationship))
                        
            neighbors.append((current_index, -round(total_weight, 4)))
            d_adj_list[current_index] = neighbors

    return d_adj_list


# Example: Adjacency list with darkness differences for a 5x5 image
width, height = 5, 5
picture = [
    0, 0.5, 0.5, 0.5, 0,
    0, 0.5, 0.5, 0.5, 0,
    0, 0, 0.8, 0, 0,
    0.2, 0.2, 0.8, 0.2, 0.2,
    1, 1, 1, 1, 1
]


# Create D_adj_list
d_adj_list = create_d_adj_list(picture, width, height)


def matvec(v, adj_list, size):
    """
    Matrix-vector multiplication using the adjacency list.
    """
    result = np.zeros(size)
    for i in range(size):
        for neighbor, weight in adj_list.get(i, []):
            result[i] += weight * v[neighbor]
    return result



# Size of the system (number of nodes)
size = width*height

# Define the LinearOperator
operator = LinearOperator(
    shape=(size, size),
    matvec=lambda v: matvec(v, d_adj_list, size)
)

k=5 #k is not number of clusters, it is the smallest k eigens (including 0)

# Compute the smallest k eigenvalue
eigenvalues, eigenvectors = eigsh(operator, k=k, which='SM', tol=1e-6)

# Print the results
for i in range(k):
    print(f'eigenvalue {i+1}: {round(eigenvalues[i],4)}')
    print(f'eigenvector {i+1}: {np.round(eigenvectors[:, i],4)}')
    print()



'''Up till here we got the eigenvalues and vectors'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
print(f'The U matrix for {picture} is:')
# Construct U matrix for non-zero eigenvalues
U = []
num_cluster = 0
for i in range(k):
    if abs(eigenvalues[i]) >= 0.00001:  # Only consider non-zero eigenvalues
        U.append(eigenvectors[:, i])
        num_cluster += 1

U = np.array(U).T  # Transpose to align the eigenvectors as columns

print(f'The U matrix for the eigenvectors corresponding to {num_cluster} samllest non-zero eigenvalues is:')
print(U)
print()
print(f'Therefore we have:')
for i in range(size):
    print(f'y{i+1} = {U[i]}')
print()

'''K-MEANS ALGORITHM'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
# Map labels to points
labels = [f'y{i+1}' for i in range(size)]

# Create a list of all points
points = list(range(1, size + 1))

# Assigning cluster label now
# Cluster label is more useful for visualising, but not for coding
clusters_label = {}
for i in range(num_cluster): # Initial assignment, take some random points in the inital cluster
    random_point = random.choice(points)
    clusters_label[i+1] = [random_point]
    points.remove(random_point)

for point in points: # each remaining point has equal chance to be in any clusters
    assign_num = random.randint(1, 100*num_cluster)
    key = assign_num%num_cluster
    if key == 0:
        key += num_cluster
    clusters_label[key].append(point)

print(clusters_label)
print()

# Actual clsuters:
clusters = {}
for i in range(num_cluster):
    clusters[i+1] = []
    for key in clusters_label[i+1]:
        clusters[i+1].append(U[key-1])

# calculate COM now
# takes in cluster labels and their point
# output the COM dictionary with key of the cluster, value is the COM of the cluster
def calculate_com(clusters_label, clusters):
    clusters_com = {}
    for i in range(len(clusters_label)):
        clusters_com[i+1] = np.mean(clusters[i+1], axis=0)
    return clusters_com

# Function to assign all point to the nearest cluster
# takes in cluster COM 
# output new labels and new clusters
def assign_to_cluster(U, clusters_com):
    new_clusters = {}
    new_clusters_label = {}
    for key in range(len(clusters_com)):
        new_clusters[key+1] = []
        new_clusters_label[key+1] = []
    
    
    for i in range(size):
        min_dis = np.linalg.norm(U[i] - clusters_com[1])
        new_cluster = 1
        
        for j in range(len(clusters_com)):
            distance = np.linalg.norm(U[i] - clusters_com[j+1])
            if distance < min_dis:
                min_dis, new_cluster = distance, j+1
        
        new_clusters_label[new_cluster].append(i+1)
        new_clusters[new_cluster].append(U[i])
        
    return new_clusters_label, new_clusters

'''Iterating assignments till stablized'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
# Iterative K-means loop
iteration = 0
previous_clusters_label = None

while True and iteration < 1000: 
    iteration += 1
    print(f"\nIteration {iteration}:")
    
    # Calculate the center of mass for each cluster
    clusters_com = calculate_com(clusters_label, clusters)
    
    # Reassign points to the nearest cluster
    new_clusters_label, new_clusters = assign_to_cluster(U, clusters_com)
    
    # Print the cluster assignment for the current iteration
    for i in range(len(new_clusters_label)):
        print(f"C{i+1} has: {new_clusters_label[i+1]}")
    
    # Check for convergence
    if new_clusters_label == previous_clusters_label:
        print("\nClusters stabilized. Stopping iteration.")
        break
    
    # Update the previous cluster labels and clusters
    previous_clusters_label = clusters_label
    clusters_label = new_clusters_label
    clusters = new_clusters

    

# Final cluster results
print("\nFinal Clusters:")
for i in range(len(clusters_label)):
        print(f"C{i+1} has: {clusters_label[i+1]}")
        
        
'''Up till here, we fully obtained the clusters'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------'''
# Visualising next

# Plotting the grid
# Reshape the picture array into a 5x5 matrix
picture = np.array(picture)
picture_reshaped = picture.reshape(height, width)

# Plotting the 5x5 grid
fig = plt.figure(figsize=(8, 8))
plt.imshow(picture_reshaped, cmap='gray_r', interpolation='nearest', vmin=0, vmax=1)  # Ensure vmin and vmax are set
plt.colorbar(label='Darkness Level (0: White, 1: Black)')
plt.title("5x5 Darkness Grid (0: White, 1: Black)")
plt.axis('off')  # Turn off the axis

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
    "#fd8d3c", "#74c476", "#9e9ac8", "#969696", "#6baed6", "#fdbe85", "#a1d99b", "#bcbddc", "#d9d9d9", "#9ecae1",
    "#fdd0a2", "#c7e9c0", "#dadaeb", "#f7f7f7", "#c6dbef", "#fee6ce", "#e5f5e0", "#edf8fb", "#deebf7", "#fff5eb",
    "#f7fcf5", "#f7fcfd", "#eff3ff", "#fee0d2", "#a50f15", "#d73027", "#fee08b", "#91bfdb", "#4575b4", "#e41a1c",
    "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3", "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
    "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#bc80bd", "#ccebc5", "#ffed6f", "#9b59b6", "#2ecc71", "#3498db",
    "#e74c3c", "#ecf0f1", "#95a5a6", "#34495e", "#16a085", "#f39c12", "#c0392b", "#7f8c8d"]  # Extend as needed for more clusters

# Get the figure size in inches
fig_width, fig_height = fig.get_size_inches()

for i in range(picture_reshaped.shape[0]):  # Iterate over rows
    for j in range(picture_reshaped.shape[1]):  # Iterate over columns
        # Place a number on each grid cell
        point_index = i * picture_reshaped.shape[1] + j + 1
        
        # Find the cluster for the current point_index
        cluster_number = None
        for cluster, points in clusters_label.items():
            if point_index in points:
                cluster_number = cluster
                break
            
        # Get the color for the cluster
        cluster_color = colors[cluster_number - 1]  # cluster_number is 1-based
        
        plt.text(j, i, str(point_index), 
                 ha='center', va='center', color=cluster_color, fontsize=fig_width*5)
        
# Add the side legend for clusters (on the left)
# Adjusted the position to ensure the plot and the legend do not overlap
fig.subplots_adjust(left=0.2)  # Make space on the left for the legend

legend_ax = fig.add_axes([0.05, 0.1, 0.15, 0.8])  # Positioning the legend axis to the left
legend_ax.axis('off')  # Hide axis for the legend

# Display the cluster colors and labels in the legend panel
for idx, (cluster, color) in enumerate(zip(clusters_label.keys(), colors)):
    legend_ax.text(0.1, 1 - 0.1 * idx, f'Cluster {cluster}', color=color, fontsize=fig_width*2, ha='left', va='top')

plt.show()