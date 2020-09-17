import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 32

train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                        train=True, 
                                        download=True, 
                                        transform=transforms.Compose([
                                        transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                        train=False, 
                                        download=True, 
                                        transform=transforms.Compose([
                                        transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, 
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)

# Just printing an image from the dataset
# image, label = next(iter(train_set))
# plt.imshow(image.view(28,28), cmap="gray")
# plt.show()
# print(label)

class Fashion_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d((2,2))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d((2,2))
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

NN = Fashion_MNIST()

optimizer = optim.Adam(NN.parameters(), lr=1e-3)
loss_function = nn.NLLLoss()

EPOCHS = 5

for epoch in range(0,EPOCHS):
    for data in train_loader:
        X, y = data
        NN.zero_grad()
        output = NN(X.view(-1, 1, 28, 28))
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad(): 
    for data in test_loader:
        X, y = data
        NN.zero_grad()
        output = NN(X.view(-1, 1, 28, 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))