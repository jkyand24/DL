import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg11, resnet18
from sklearn.metrics import accuracy_score

#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(root="./data", train=True, download=False,
                        transform=train_transform)
test_dataset = CIFAR10(root="./data", train=False, download=False,
                       transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2,
                          pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2,
                         pin_memory=True)

# 여러 model의 출력을 평균하여 출력

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()

        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]

        outputs = torch.stack(outputs, dim=0)

        outputs = torch.mean(outputs, dim=0)

        return outputs

#

def train(train_loader, model, device, criterion, optimizer):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

#

def evaluate(test_loader, model, device):
    model.eval()

    prediction_list = []
    target_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            _, prediction = torch.max(output, 1)

            prediction_list.extend(prediction.cpu().numpy())

            target_list.extend(target.cpu().numpy())

    acc = accuracy_score(target_list, prediction_list)

    return acc


#

vgg_model = vgg11(pretrained=True)
resnet_model = resnet18(pretrained=True)

num_features_vgg = vgg_model.classifier[6].in_features
num_features_resnet = resnet_model.fc.in_features

vgg_model.classifier[6] = nn.Linear(num_features_vgg, 10)
resnet_model.fc = nn.Linear(num_features_resnet, 10)

ensemble_model = EnsembleModel([vgg_model, resnet_model])

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(ensemble_model.parameters(), lr=0.001)

num_epochs = 5

#

if __name__ == "__main__":
    for epoch in range(num_epochs):
        print(f"\nEpoch # {epoch + 1}")

        ensemble_model = ensemble_model.to(device)

        print("\nTraining............................")

        train(train_loader, ensemble_model, device, criterion, optimizer)

        print("\nEvaluation............................")

        acc = evaluate(test_loader, ensemble_model, device)

        print(f"Accuracy: {acc:.2f}")