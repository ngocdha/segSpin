from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

img = Image.open("mario.jpg")
img_gray = img.convert("L")
img_array = np.array(img_gray)

# Break into 3x3 blocks
blocks = img_array.reshape(4, 4, 4, 4).swapaxes(1, 2)
block_means = np.array([[np.mean(block) for block in row] for row in blocks])

plt.imshow(block_means, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Value')  # add color scale
plt.title('Heatmap of 2D Array')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()

print(block_means)
