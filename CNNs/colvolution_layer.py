import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image_path = "./data/conv2d_image.jpg"
image = Image.open(image_path).convert('L')
input_data = torch.unsqueeze(torch.from_numpy(np.array(image)), dim=0).float()

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2)

output_data = conv(input_data)

plt.subplot(1, 2, 1)
plt.imshow(input_data.squeeze(), cmap='gray')
plt.title('input')

plt.subplot(1, 2, 2)
plt.imshow(output_data.squeeze().detach().numpy(), cmap='gray')
plt.title('Output')

plt.tight_layout()
plt.show()