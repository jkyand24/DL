import torch
import torch.nn as nn

#

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
    
input_size = 784
hidden_size = 256
output_size = 10

model = ANN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # optimizer에 모델의 모든 parameter 등록됨

#

inputs = torch.randn(100, input_size) # 100 * input_size
labels = torch.randint(0, output_size, (100, )) # 100

#

num_epochs = 10

for epoch in range(num_epochs):
    outputs = model(inputs)
    
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    
    loss.backward() # 역전파 이후, Autograd가 parameter의 .grad 속성에 각 parameter에 대한 gradient를 계산하고 저장
    
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")