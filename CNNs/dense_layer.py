import torch
import torch.nn as nn
import matplotlib.pyplot as plt

input_size = 4
output_size = 2

dense_layer = nn.Linear(input_size, output_size)

weights = dense_layer.weight.detach().numpy()

plt.imshow(weights, cmap='coolwarm', aspect='auto')
plt.xlabel("Input Features")
plt.ylabel("Output")
plt.title("Dense Layer Weights")
plt.colorbar()
plt.show()