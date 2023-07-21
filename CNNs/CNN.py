import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # (?, 28, 28, 1) -> (?, 28, 28, 32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
        self.conv1_output = None
        self.conv2_output = None
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        self.conv1_output = x
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        self.conv2_output = x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
# data set, data loader 준비

train_dataset = MNIST(root = "./data", train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root = "./data", train=False, transform=ToTensor())

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# model, loss function, optimizer 준비

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training

num_epochs = 4

loss_per_epoch_list = []

fig, axes =plt.subplots(2, 3, figsize=(10, 8))

fig.tight_layout(pad=4.0)

axes = axes.flatten()

for epoch in range(num_epochs):
    # Training 
    
    model.train()
    
    loss_per_epoch = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        loss_per_epoch += loss.item()
    
    print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss_per_epoch:.4f}")
    
    loss_per_epoch_list.append(loss_per_epoch)
    
    # 시각화
    
    if epoch == 0:
        # 
        
        weights = model.conv1.weight.detach().cpu().numpy()
        
        img = axes[0].imshow(weights[0, 0], cmap='coolwarm')
    
        axes[0].set_title('Conv1 Weights')
        
        divider = make_axes_locatable(axes[0])
        
        cax = divider.append_axes('right', size='5%', pad=0.05)
        
        plt.colorbar(img, cax=cax)
        
        #
        
        weights = model.conv2.weight.detach().cpu().numpy()
        
        img = axes[1].imshow(weights[0, 0], cmap='coolwarm')
    
        axes[1].set_title('Conv2 Weights')
        
        divider = make_axes_locatable(axes[1])
        
        cax = divider.append_axes('right', size='5%', pad=0.05)
        
        plt.colorbar(img, cax=cax)
        
        #
        
        if model.conv1_output is not None:
            conv1_output = model.conv1_output.detach().cpu().numpy()
            
            img = axes[2].imshow(conv1_output[0, 0], cmap='coolwarm')
        
            axes[2].set_title('Conv1 Output')
            
            divider = make_axes_locatable(axes[2])
            
            cax = divider.append_axes('right', size='5%', pad=0.05)
            
            plt.colorbar(img, cax=cax)
        
        #
        
        if model.conv2_output is not None:
            conv2_output = model.conv2_output.detach().cpu().numpy()
            
            img = axes[3].imshow(conv2_output[0, 0], cmap='coolwarm')
        
            axes[3].set_title('Conv2 Output')
            
            divider = make_axes_locatable(axes[3])
            
            cax = divider.append_axes('right', size='5%', pad=0.05)
            
            plt.colorbar(img, cax=cax)
        
        #
        
axes[4].plot(range(num_epochs), loss_per_epoch_list)

axes[4].set_title('Training Loss')

axes[4].set_xlabel('Epoch')

axes[4].set_xlabel('Loss')
        
plt.show() 
        
# Evaluation

model.eval()

correct = 0 
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
accuracy = 100.0 * (correct / total)

print(f"Test Accuracy: {accuracy:.2f}%")