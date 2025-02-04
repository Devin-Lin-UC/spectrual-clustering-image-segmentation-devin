import sys
import subprocess

# List of required packages
required_packages = [
    "numpy", "scipy", "matplotlib", "PILLOW"
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
# image_path = "https://raw.githubusercontent.com/Devin-Lin-UC/spectrual-clustering-image-segmentation-devin/refs/heads/main/images/tree.jpg"
response = requests.get(image_path)
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
else:
    print(f"Error: Unable to fetch image, status code {response.status_code}")


# Convert to grayscale
grayscale_image = image.convert("L")

# Resize to a square 20x20 grid, maintaining aspect ratio for the minimalist effect
resized_image = grayscale_image.resize((width, height), Image.Resampling.NEAREST)

# Normalize pixel values to a 0-1 scale
pixel_array = 1-np.array(resized_image) / 255.0

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
            current_value = picture[current_index]  # Darkness value of the current pixel
            
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
                        neighbor_value = picture[neighbor_index]
                        
                        '''Note you can manupulate the 28 since it's the theta parameter'''
                        '''----------------------------------------------------------------------------------------------------'''
                        relationship = 10**( -28*(current_value - neighbor_value)**2)
                        total_weight += relationship
                        neighbors.append((neighbor_index, relationship))
                        
            neighbors.append((current_index, -round(total_weight, 16)))
            d_adj_list[current_index] = neighbors

    return d_adj_list



picture = pixel_array.ravel().tolist()


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


# Create a histogram of pixel darkness values
hist, bins = np.histogram(picture, bins=50, range=(0, 1))

# Find peaks in the histogram (local maxima)
peaks, _ = find_peaks(hist)

# Get the heights of the peaks
peak_heights = hist[peaks]
# Sort peaks in descending order to process from largest to smallest
sorted_peak_heights = sorted(zip(peak_heights, peaks), reverse=True)
peak_heights = np.array(sorted_peak_heights)


# Define a function to check if a peak satisfies the cluster criteria
def check_cluster(peak, prev_peak, minimum, ratio=0.05**4):
    # The threshold is either proportional to the previous peak or based on the first peak
    threshold = min(np.sqrt(np.sqrt(ratio)) * prev_peak, minimum)
    return peak >= threshold

# Initialize the number of clusters
clusters = 0
prev_peak = peak_heights[0][0]
minimum = peak_heights[0][0]*0.1
clusters += 1  # The first peak is always considered a cluster

# Check for subsequent peaks based on relative drops
for i in range(1, len(peak_heights)):
    current_peak = peak_heights[i][0]
    
    if i == 1:  # The second peak is based on the first
        if check_cluster(current_peak, prev_peak, minimum):
            clusters += 1
            prev_peak = current_peak

    elif i == 2:  # The third peak is based on both the first and second peaks
        if check_cluster(current_peak, prev_peak, minimum, ratio=0.3):
            clusters += 1
            prev_peak = current_peak

            
    elif i == 3:  # The fourth peak is based on the third and first peaks
        if check_cluster(current_peak, prev_peak, minimum, ratio=0.4):
            clusters += 1
            prev_peak = current_peak

    else:
        # Continue for further peaks with similar logic
        if check_cluster(current_peak, prev_peak, minimum, ratio=0.5):
            clusters += 1
            prev_peak = current_peak


# Output the number of clusters detected
print(f'Number of clusters detected: {clusters}')


'''the k is defined here automatically, you can uncomment the cluster line below'''
'''----------------------------------------------------------------------------------------------------'''
# clusters = the number you want to define manually
k=clusters+1 #k is not number of clusters, it is the smallest k eigens (including 0)


# Compute the smallest k eigenvalue
eigenvalues, eigenvectors = eigsh(operator, k=k, which='SM', tol=1e-6)

# Print the results
for i in range(k):
    print(f'eigenvalue {i+1}: {round(eigenvalues[i],6)}')
    print(f'eigenvector {i+1}: {np.round(eigenvectors[:, i],6)}')
    print()



'''Up till here we got the eigenvalues and vectors'''

# print(f'The U matrix for {picture} is:')
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
    
    
    # Check for convergence
    if new_clusters_label == previous_clusters_label:
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
picture_reshaped = picture.reshape(height, width)

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


# Create the figure and set up the subplots (1 row, 3 columns)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 1. Plot the Histogram
axs[0].hist(picture, bins=50, range=(0, 1), alpha=0.7, color='blue', edgecolor='black')
axs[0].set_title("Histogram of Darkness Values")
axs[0].set_xlabel("Pixel Darkness (0 to 1)")
axs[0].set_ylabel("Frequency")

# 2. Plot the original image or graph (e.g., reshaped image)
axs[1].imshow(picture_reshaped, cmap="gray")
axs[1].set_title("Original Image (Grayscale)")
axs[1].axis('off')  # Hide axes for clarity

# 3. Plot the Contour Grid (with clusters)
x = np.linspace(0, width, width+1) 
y = np.linspace(0, height, height+1) 

# Create 2D grids for x and y
X, Y = np.meshgrid(x, y)  # Shape of X and Y will both be (width, height)
contour_grid = np.zeros_like(picture_reshaped, dtype=int)


# Fill in the contour grid with the cluster labels
for cluster, points in clusters_label.items():
    for point in points:
        row = (point - 1) // height  # Calculate row index
        col = (point - 1) % width  # Calculate column index
        contour_grid[-(row+1), col] = cluster  # Assign cluster number

# Set up the colormap with your custom colors
cmap = ListedColormap(colors[:k-1])

# Ensure discrete levels are used (match the number of clusters)
contour_levels = np.arange(0, len(clusters_label) + 1)  # Make sure this aligns with your cluster values

# Now plot the contour using your colormap
contour = axs[2].pcolormesh(X, Y, contour_grid, cmap=cmap, shading = 'flat')

# Add a colorbar
fig.colorbar(contour, ax=axs[2], ticks=range(1, len(clusters_label) + 1), label='Cluster Number')

axs[2].set_title("Contour Grid by Cluster")
axs[2].axis('off')  # Hide axes for clarity

# Show the plots
plt.tight_layout()  # Automatically adjust subplot parameters for better layout
plt.show()
