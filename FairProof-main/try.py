from sklearn.decomposition import PCA
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Define a simple neural network with 4 input features and 2 hidden neurons


def neural_network_4d(x1, x2, x3, x4):
    # Weights and biases for the hidden layer
    # 2x4 weights for hidden layer
    W1 = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]])
    b1 = np.array([0.5, -0.5])                      # 2 biases for hidden layer
    # Weights and bias for the output layer
    # 2 weights for output layer
    W2 = np.array([1, 1])
    b2 = -0.5                                       # 1 bias for output layer

    # Hidden layer activations (ReLU)
    z1 = np.maximum(0, W1[0, 0] * x1 + W1[0, 1] * x2 +
                    W1[0, 2] * x3 + W1[0, 3] * x4 + b1[0])
    z2 = np.maximum(0, W1[1, 0] * x1 + W1[1, 1] * x2 +
                    W1[1, 2] * x3 + W1[1, 3] * x4 + b1[1])

    # Output layer activation (linear)
    output = W2[0] * z1 + W2[1] * z2 + b2

    return output


# Create a grid of points in 4D
x1_range = np.linspace(-2, 2, 20)
x2_range = np.linspace(-2, 2, 20)
x3_range = np.linspace(-2, 2, 20)
x4_range = np.linspace(-2, 2, 20)
x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(
    x1_range, x2_range, x3_range, x4_range)
points_4d = np.array([x1_grid.flatten(), x2_grid.flatten(),
                     x3_grid.flatten(), x4_grid.flatten()]).T
outputs = neural_network_4d(
    points_4d[:, 0], points_4d[:, 1], points_4d[:, 2], points_4d[:, 3])

# Project 4D points to 3D using PCA
pca = PCA(n_components=3)
points_3d = pca.fit_transform(points_4d)

# Plot the decision boundary
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                c=outputs.flatten(), cmap='coolwarm', alpha=0.5)
plt.colorbar(sc, label='Output Value')
ax.set_title("4D to 3D Projection with Color Representing 4th Dimension")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()
