import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, visible_size, hidden_size, k):
        super(RBM, self).__init__()
        
        self.W = nn.Parameter(torch.randn(hidden_size, visible_size) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(visible_size))
        self.h_bias = nn.Parameter(torch.zeros(hidden_size))
        self.k = k
    
    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size())))) # by subtracting Variable(torch.rand(p.size())), noise introduced
    
    def v_to_h(self, v):
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        
        sample_h = self.sample_from_p(p_h)
        
        return sample_h
    
    def h_to_v(self, h):
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        
        sample_v = self.sample_from_p(p_v)
        
        return sample_v
    
    def forward(self, v):
        h_ = self.v_to_h(v)
        
        for _ in range(self.k):
            v_ = self.h_to_v(h_)
            h_ = self.v_to_h(v_)
            
        return v_
        
    def free_energy(self, v): # 수식에 따라 그대로 작성됨
        v_v_bias = v.mv(self.v_bias)
        
        v_W_h_bias = F.linear(v, self.W, self.h_bias)
        hidden_term = v_W_h_bias.exp().add(1).log().sum(1)
        
        return (- hidden_term - v_v_bias).mean()

#

transform = transforms.Compose([transforms.ToTensor(),
                                ])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#

visible_size = 784
hidden_size = 256

rbm = RBM(visible_size, hidden_size, 1)

optimizer = torch.optim.SGD(rbm.parameters(), lr=0.1)

#

num_epochs = 10

for epoch in range(num_epochs):
    #
    
    for images, _ in train_dataloader:
        
        inputs = Variable(images.view(-1, visible_size))
        
        sample_inputs = inputs.bernoulli()
        
        outputs = rbm(sample_inputs)
        
        loss = rbm.free_energy(sample_inputs) - rbm.free_energy(outputs)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
    
    #
    
    vutils.save_image(rbm.W.view(hidden_size, 1, 28, 28), f"./result2/Weights_epoch{epoch+1}.png", normalize=True)
    
    inputs_display = inputs.view(-1, 1, 28, 28)
    outputs_display = outputs.view(-1, 1, 28, 28)
    
    comparison = torch.cat([inputs_display, outputs_display], dim=3)
    
    vutils.save_image(comparison, f"./result2/Reconstruction_epoch{epoch+1}.png", normalize=True)