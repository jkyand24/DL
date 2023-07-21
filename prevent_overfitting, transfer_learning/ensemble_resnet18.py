import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#

train_transform = transform.Compose([
    transform.RandomHorizontalFlip(),
    transform.RandomVerticalFlip(),
    transform.RandAugment(),
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 -1 ~ 1로 정규화
])
test_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 -1 ~ 1로 정규화
])

train_dataset = CIFAR10(root="./data", train=True, download=True,
                        transform=train_transform)
test_dataset = CIFAR10(root="./data", train=False, download=True,
                       transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=True)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.001)


#

def train(device, train_loader, model, criterion, optimizer):
    model.train()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        outputs = model(data)

        loss = criterion(outputs, target)

        loss.backward()

        optimizer.step()


#

def evaluate(device, test_loader, model):
    model.eval()

    preds_list = []
    target_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)

            _, preds = torch.max(outputs, 1)

            preds_list.extend(preds.cpu().numpy())
            target_list.extend(target.cpu().numpy())

    acc = accuracy_score(target_list, preds_list)

    return acc


#

def ensemble_predict(device, models, test_loader):
    preds_list = []

    with torch.no_grad():
        for data, _ in test_loader:  # 한 batch씩
            data = data.to(device)

            outputs_list = []

            for model in models:  # 한 model씩
                model = model.to(device)

                model.eval()

                outputs = model(data)

                outputs_list.append(outputs)

            # voting

            ensemble_outputs = torch.stack(outputs_list).mean(
                dim=0)  # 각 model이 낸 결과의 평균. ensemble_outputs.size(): [batch_size, num_classes]
            # torch.stack() --> [num_epochs, batch_size, num_classes]

            _, preds = torch.max(ensemble_outputs, 1)  # preds: 한 batch에서의 prediction

            preds_list.extend(preds.cpu().numpy())

    return preds_list


#

if __name__ == "__main__":
    models = []

    model = model.to(device)

    num_models = 3

    num_epochs = 2

    # epoch가 끝날 때마다 models에 append
    # => models[epoch]: epoch번째의 결과 model

    for i in range(num_models):
        # i번째 model을 num_epochs회 만큼 반복 Training & Evaluation.

        for epoch in range(num_epochs):
            print(f"\nTraining, Epoch # {epoch + 1}--------------")

            train(device, train_loader, model, criterion, optimizer)

            print(f"\nEvaluation, Epoch # {epoch + 1}--------------")

            acc = evaluate(device, test_loader, model)

            print(f"Accuracy: {acc:.2f}")

        # i번째 model의 train 완료된 결과를 models에 append.

        models.append(model)

    ensemble_predictions = ensemble_predict(device, models, test_loader)

    ensemble_acc = accuracy_score(test_dataset.targets, ensemble_predictions)

    print(f"\nEnsemble Accuracy: {ensemble_acc:.2f}")