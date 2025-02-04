from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Example: Adjacency list with darkness differences for a 5x5 image
width, height = 300, 300

# image_path = "E:/UC stuff/spectrual clustering/tree.jpg"
image_path = 'E:/UC stuff/spectrual clustering/dst_base.png'
image = Image.open(image_path)

# Resize the image to 100x100 pixels
resized_image = image.resize((width, height), Image.Resampling.NEAREST)

# Convert the image to an array
pixel_array = np.array(resized_image)

# Normalize RGB pixel values to a 0-1 scale (for each channel)
normalized_rgb = pixel_array / 255.0

# Get the individual RGB channels
red_channel = normalized_rgb[:, :, 0]  # Red channel
green_channel = normalized_rgb[:, :, 1]  # Green channel
blue_channel = normalized_rgb[:, :, 2]  # Blue channel

print(normalized_rgb)
print()
print(red_channel)
print()
print(green_channel)
print()
print(blue_channel)
print()

# Plot the resized image
plt.imshow(resized_image)
plt.title("Resized Image")
plt.axis('off')  # Turn off axis labels and ticks
plt.show()