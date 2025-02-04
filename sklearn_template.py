import numpy as np
import math
import random  # Import random module
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import euclidean_distances


# 100 data points
data_points = [
 [6.2483570765056164, 5.930867849414407],
 [6.323844269050346, 6.761514928204012],
 [5.882923312638332, 5.88293152152541],
 [6.789606407753696, 6.383717364576454],
 [5.765262807032524, 6.2712800217929825],
 [5.768291153593768, 5.767135123214872],
 [6.120981135783017, 5.043359877671101],
 [5.137541083743484, 5.718856235379514],
 [5.493584439832788, 6.157123666297637],
 [5.545987962239394, 5.293848149332354],
 [6.732824384460777, 5.8871118497567325],
 [6.033764102343962, 5.287625906893272],
 [5.727808637737409, 6.055461294854933],
 [5.424503211288848, 6.187849009172836],
 [5.699680655040598, 5.854153125103362],
 [5.699146693885302, 6.926139092254469],
 [5.993251387631033, 5.47114453552205],
 [6.411272456051594, 5.389578175014488],
 [6.104431797502378, 5.0201649380601125],
 [5.335906975550785, 6.098430617934562],
 [8.369233289997705, 4.085684140594985],
 [7.9421758588058795, 3.8494481522053556],
 [7.260739004816286, 3.6400778958026456],
 [7.769680614520106, 4.528561113109458],
 [8.17180914478423, 3.118479922318633],
 [8.162041984697398, 3.8074588597918417],
 [7.6615389998470205, 4.305838144420434],
 [8.515499761247975, 4.465640059558099],
 [7.580391238388681, 3.8453938120743927],
 [8.165631715701782, 4.48777256356118],
 [7.760412881077355, 3.9071705116680913],
 [7.446832512996986, 3.4018966879596646],
 [8.4062629111971, 4.678120014285412],
 [7.963994939209833, 4.501766448946012],
 [8.180818012523817, 3.677440122697438],
 [8.180697802754207, 4.769018283232985],
 [7.9820869804450245, 4.782321827907003],
 [6.690127447955128, 4.410951252187612],
 [8.043523534119085, 3.8504963247670663],
 [8.04588038826775, 3.006215542699554],
 [2.890164056081244, 8.178556285755873],
 [3.738947022370758, 7.740864890863176],
 [2.5957531985534064, 7.749121478207732],
 [3.457701058851037, 8.164375554829842],
 [2.735119898116481, 8.256633716556678],
 [3.0485387746740202, 8.484322495266445],
 [2.648973453061324, 7.836168926701116],
 [2.803945923433921, 7.268242525933941],
 [3.148060138532288, 8.130527636089944],
 [3.0025567283212307, 7.882706433312427],
 [2.292314628974793, 7.78967733861732],
 [2.828642741736615, 7.59886136538919],
 [2.9193571441669954, 8.20202542840727],
 [3.943092950605265, 8.08728890641592],
 [3.1287751953613823, 7.962777042116916],
 [2.0406143923504794, 7.986743062275392],
 [3.030115104970513, 9.231621056242643],
 [2.903819517609439, 8.150773671166807],
 [2.9826441151473784, 7.415660981190234],
 [3.5714114072575103, 8.375966516343388],
 [5.3955159735215235, 4.54530627260263],
 [5.701397155468049, 4.29907446860386],
 [5.293428546900135, 6.095227812904989],
 [4.504731837434655, 4.716851135198614],
 [5.049825682543821, 4.7482621729419],
 [4.224668284466934, 5.034281487403014],
 [4.468848143136947, 5.23679621531759],
 [4.540287882883098, 5.77496720250877],
 [4.608373353831881, 4.838969241897162],
 [5.406758608684835, 4.3845678417830225],
 [5.113729967302064, 5.653571377141214],
 [4.196258382719386, 5.092316929266152],
 [5.129941397124211, 5.390911435888655],
 [4.381524644560959, 4.339771693457862],
 [5.260970782808449, 5.148492336616593],
 [5.125246425172938, 5.1732241047484875],
 [4.659987639210755, 5.116126848580501],
 [5.14653623664934, 4.642824290986816],
 [5.932887255572378, 5.236916460455894],
 [4.404348251398676, 5.328276804316915],
 [6.5126591648863394, 7.393542301871226],
 [7.579297789503702, 6.5896588408241445],
 [7.481688064622161, 7.206390463468249],
 [7.411030079997245, 7.948396491326974],
 [6.877305941998565, 6.623131917821255],
 [6.555242785187239, 6.592094857517281],
 [6.961449145292948, 7.170575987408322],
 [7.13834539966501, 7.413591624518012],
 [7.006500945938954, 7.726767038578658],
 [6.867671583381022, 8.36008458329481],
 [7.312833673882503, 6.571421221791859],
 [6.4645537509694435, 7.241236207621593],
 [6.888268607337075, 7.357000247046046],
 [7.236618812286772, 6.963585543671564],
 [6.576603140965798, 6.242576387657068],
 [6.776742523966489, 7.428199397161736],
 [7.107046872065102, 6.377130610644006],
 [7.086590462925591, 7.192658689864419],
 [6.558071281899434, 7.076862552972764],
 [7.029104359223, 6.428514851084689]
]



n = len(data_points)

# Compute pairwise distances
distances = euclidean_distances(data_points)
mean_distance = np.mean(distances)
median_distance = np.median(distances)

# Suggested gamma values
gamma_value = 1 / mean_distance**2
print("Test gamma values:", gamma_value)


n_clusters = 8  # Number of clusters

# Use Spectral Embedding from sklearn for Laplacian Eigenmap
embedding = SpectralEmbedding(n_components=2, affinity='rbf', gamma=1/1000*gamma_value)   # small gamma = loose cluster, large gamma = tight clusters
U = embedding.fit_transform(data_points)


'''K-MEANS ALGORITHM'''
# Map labels to points


# Use KMeans to initialize clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_kmeans = kmeans.fit_predict(U)  # Fit on transformed data (U)


# Initial cluster assignments
clusters = {i: [] for i in range(n_clusters)}
for i, label in enumerate(labels_kmeans):
    clusters[label].append(i)




# Print the results
for item in clusters:
    print(f"Cluster C{item+1}: {clusters[item]}")
print()


# Function to calculate the center of mass for a cluster
def calculate_com(cluster_indices, data):
    points = [data[i] for i in cluster_indices]
    return np.mean(points, axis=0)

# Function to assign points to the nearest cluster
def assign_to_nearest_cluster(point, cluster_coms):
    distances = [np.linalg.norm(point - com) for com in cluster_coms]
    return np.argmin(distances)

# Iterative K-means loop
iteration = 0

while True:
    iteration += 1
    print(f"\nIteration {iteration}:")
    
    # Calculate centers of mass for all clusters
    cluster_coms = [calculate_com(cluster, U) for cluster in clusters.values()]
    print(f"Cluster centers of mass: {cluster_coms}")
    
    # Reassign points to nearest clusters
    new_clusters = {i: [] for i in range(n_clusters)}
    for i, point in enumerate(U):
        nearest_cluster = assign_to_nearest_cluster(point, cluster_coms)
        new_clusters[nearest_cluster].append(i)
    
    # Check if clusters have stabilized
    if all(set(new_clusters[i]) == set(clusters[i]) for i in range(n_clusters)):
        print("Clusters have stabilized!")
        break

    # Update clusters for the next iteration
    clusters = new_clusters

# Final cluster results
print(iteration)






# Plot results
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
plt.figure(figsize=(12, 6))

# Original space plot
for cluster_idx, cluster_indices in clusters.items():
    cluster_points = [data_points[i] for i in cluster_indices]
    x, y = zip(*cluster_points)
    plt.scatter(x, y, color=colors[cluster_idx], label=f'Cluster {cluster_idx+1}')
    com = calculate_com(cluster_indices, data_points)
    plt.scatter(com[0], com[1], color=colors[cluster_idx], marker='x', s=100, label=f'COM Cluster {cluster_idx+1}')


# Display the plot
plt.legend()
plt.title('K-means Clusters with Centers of Mass (Original Space)')
plt.xlabel('X')
plt.ylabel('Y')

# Now, plot the transformed space (y1, y2) graph (Graph 2)
# Use the first two components of U (already in the transformed space)
transformed_data = U

# Create the second plot (transformed data points)
plt.figure(figsize=(12, 6))

for cluster_idx, cluster_indices in clusters.items():
    cluster_points = [U[i] for i in cluster_indices]
    x, y = zip(*cluster_points)
    plt.scatter(x, y, color=colors[cluster_idx], label=f'Cluster {cluster_idx+1}')
    com = cluster_coms[cluster_idx]
    plt.scatter(com[0], com[1], color=colors[cluster_idx], marker='x', s=100, label=f'COM Cluster {cluster_idx+1}')

# Display the plot
plt.legend()
plt.title('Transformed K-means Clusters with Centers of Mass')
plt.xlabel('y1')
plt.ylabel('y2')

# Show both plots
plt.show()