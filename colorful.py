import sys
import subprocess

# List of required packages
required_packages = [
    "numpy", "scipy", "matplotlib", "PILLOW", 'requests'
]

# Check and install missing packages
for package in required_packages:
    try:
        __import__(package if package != "PILLOW" else "PIL")
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import numpy as np
import math
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
import random
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap
from scipy.signal import find_peaks


print('start running')
print('please wait for a minute or two')

'''don't worry about the codes, I would use --- to highlight what you can change around'''
'''----------------------------------------------------------------------------------------------------'''



'''choose your dimension, preferably not over 80 by 80'''
'''----------------------------------------------------------------------------------------------------'''
width, height = 80, 80



import requests
from PIL import Image
from io import BytesIO


'''uncomment the image you want to cluster'''
'''----------------------------------------------------------------------------------------------------'''
image_path = "https://raw.githubusercontent.com/Devin-Lin-UC/spectrual-clustering-image-segmentation-devin/refs/heads/main/images/house.jpg"
# image_path = "https://raw.githubusercontent.com/Devin-Lin-UC/spectrual-clustering-image-segmentation-devin/refs/heads/main/images/moon_tree.jpg"
# image_path = "https://raw.githubusercontent.com/Devin-Lin-UC/spectrual-clustering-image-segmentation-devin/refs/heads/main/images/mountain.jpg"
# image_path = "https://raw.githubusercontent.com/Devin-Lin-UC/spectrual-clustering-image-segmentation-devin/refs/heads/main/images/pfp.jpg"
image_path = "https://raw.githubusercontent.com/Devin-Lin-UC/spectrual-clustering-image-segmentation-devin/refs/heads/main/images/tree.jpg"
response = requests.get(image_path)
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
else:
    print(f"Error: Unable to fetch image, status code {response.status_code}")


# Resize the image to 100x100 pixels
resized_image = image.resize((width, height), Image.Resampling.NEAREST)

# Convert the image to an array
pixel_array = np.array(resized_image)/ 255.0
# Get the individual RGB channels
red = pixel_array[:, :, 0]  # Red channel
green = pixel_array[:, :, 1]  # Green channel
blue = pixel_array[:, :, 2]  # Blue channel




def log_numbers(n):
    result = [0, 1, -1]
    boundary = 1
    
    while True:
        next_boundary = int(math.exp(boundary))+1  # Calculate exponential boundary
        if next_boundary > n:
            break
        result.append(next_boundary)
        result.append(-next_boundary)
        boundary += 1
    
    return result


def create_d_adj_list(picture, width, height):
    """
    Create an adjacency list with neighbors and their square difference in darkness.
    """
    d_adj_list = {}
    drow_option = log_numbers(width)
    dcol_option = log_numbers(height)
    
    for row in range(height):
        for col in range(width):
            
            current_index = row * width + col
            red_value = picture[current_index][0]  # red value of the current pixel
            green_value = picture[current_index][1]  # green value of the current pixel
            blue_value = picture[current_index][2]  # blue value of the current pixel
            
            
            neighbors = []
            total_weight = 0
            
            
            for drow in drow_option:
                for dcol in dcol_option:
                    if drow == 0 and dcol == 0:
                        continue
                    
                    neighbor_row = row + drow
                    neighbor_col = col + dcol
                    
                    if 0 <= neighbor_row < height and 0 <= neighbor_col < width:
                        neighbor_index = neighbor_row * width + neighbor_col
                        neighour_red_value = picture[neighbor_index][0]  # red value of the current pixel
                        neighbor_green_value = picture[neighbor_index][1]  # green value of the current pixel
                        neighbor_blue_value = picture[neighbor_index][2]  # blue value of the current pixel
                        
                        '''You can play around with the theta in the 3 relationships below if you want'''
                        '''----------------------------------------------------------------------------------------------------'''
                        red_relationship = 10**( -28*(red_value - neighour_red_value)**2)
                        green_relationship = 10**( -36*(green_value - neighbor_green_value)**2)
                        blue_relationship = 10**( -36*(blue_value - neighbor_blue_value)**2)
                        
                        relationship = (red_relationship+green_relationship+blue_relationship)/3
                        total_weight += relationship
                        neighbors.append((neighbor_index, relationship))
                        
            neighbors.append((current_index, -round(total_weight, 16)))
            d_adj_list[current_index] = neighbors

    return d_adj_list



# Create a list of RGB tuples
picture = [tuple(pixel) for row in pixel_array for pixel in row]

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



'''you can change the number of cluster'''
'''----------------------------------------------------------------------------------------------------'''
clusters = 5
k=clusters+1 #k is not number of clusters, it is the smallest k eigens (including 0)


# Compute the smallest k eigenvalue
eigenvalues, eigenvectors = eigsh(operator, k=k, which='SM', tol=1e-6)

# Print the results
for i in range(k):
    print(f'eigenvalue {i+1}: {round(eigenvalues[i],6)}')
    print(f'eigenvector {i+1}: {np.round(eigenvectors[:, i],6)}')
    print()



'''Up till here we got the eigenvalues and vectors'''

# Construct U matrix for non-zero eigenvalues
U = []
num_cluster = 0
for i in range(k):
    if abs(eigenvalues[i]) >= 0.00001:  # Only consider non-zero eigenvalues
        U.append(eigenvectors[:, i])
        num_cluster += 1

U = np.array(U).T  # Transpose to align the eigenvectors as columns


'''K-MEANS ALGORITHM'''

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
    # for i in range(len(new_clusters_label)):
    #     print(f"C{i+1} has: {new_clusters_label[i+1]}")
    
    # Check for convergence
    if new_clusters_label == previous_clusters_label:
        # print("\nClusters stabilized. Stopping iteration.")
        break
    
    # Update the previous cluster labels and clusters
    previous_clusters_label = clusters_label
    clusters_label = new_clusters_label
    clusters = new_clusters

    




        
'''Up till here, we fully obtained the clusters'''

# Visualising next

# Plotting the grid
# Reshape the picture array into a 5x5 matrix
picture = np.array(picture)
picture_reshaped = picture.reshape(height, width, 3)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
    "#fd8d3c", "#74c476", "#9e9ac8", "#969696", "#6baed6", "#fdbe85", "#a1d99b", "#bcbddc", "#d9d9d9", "#9ecae1",
    "#fdd0a2", "#c7e9c0", "#dadaeb", "#f7f7f7", "#c6dbef", "#fee6ce", "#e5f5e0", "#edf8fb", "#deebf7", "#fff5eb",
    "#f7fcf5", "#f7fcfd", "#eff3ff", "#fee0d2", "#a50f15", "#d73027", "#fee08b", "#91bfdb", "#4575b4", "#e41a1c",
    "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3", "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
    "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#bc80bd", "#ccebc5", "#ffed6f", "#9b59b6", "#2ecc71", "#3498db",
    "#e74c3c", "#ecf0f1", "#95a5a6", "#34495e", "#16a085", "#f39c12", "#c0392b", "#7f8c8d"]
# Create a colormap using the colors list
cmap = ListedColormap(colors)


# Create the figure and set up the subplots (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(11.2, 5))


# 1. Plot the original image or graph (e.g., reshaped image)
axs[0].imshow(picture_reshaped)
axs[0].set_title("Original Image")
axs[0].axis('off')  # Hide axes for clarity

# 2. Plot the Contour Grid (with clusters)
x = np.arange(width)
y = np.arange(height)

# Create 2D grids for x and y
X, Y = np.meshgrid(x, y)  # Shape of X and Y will both be (width, height)
contour_height = np.zeros((height,width))


# Fill in the contour grid with the cluster labels
for cluster, points in clusters_label.items():
    for point in points:
        row = (point - 1) // width  # Calculate row index
        col = (point - 1) % width  # Calculate column index
        contour_height[-(row+1), col] = cluster  # Assign cluster number



# Set up the colormap with your custom colors
cmap = ListedColormap(colors[:k-1])

# Ensure discrete levels are used (match the number of clusters)
contour_levels = np.arange(0, len(clusters_label) + 1)  # Make sure this aligns with your cluster values
# Now plot the contour using your colormap
contour = axs[1].pcolormesh(X,Y,contour_height, shading='nearest', cmap=cmap)

# Add a colorbar
fig.colorbar(contour, ax=axs[1], ticks=range(1, len(clusters_label) + 1), label='Cluster Number')

axs[1].set_title("Contour Grid by Cluster")
axs[1].axis('off')  # Hide axes for clarity

# Show the plots
plt.tight_layout()  # Automatically adjust subplot parameters for better layout
plt.show()


