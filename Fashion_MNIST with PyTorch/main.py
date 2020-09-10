import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
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
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.dropout = nn.Dropout(0.2)

        self._to_linear = None
        x = torch.randn(28, 28).view(-1, 1, 28, 28)
        self.conv(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def conv(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # x = self.droput(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        # x = self.droput(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        # x = self.droput(x)

        if not self._to_linear:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

NN = Fashion_MNIST()

optimizer = optim.Adam(NN.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

EPOCHS = 5

for epoch in tqdm(range(0, EPOCHS)):
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

with torch.no_grad(): #no_grad means no gradients. we get rid of the gradient to calculate accuracy. we dont wanna know gradients; we wnant to know accuracy 
    for data in train_loader:
        X, y = data
        output = NN(X.view(-1, 1, 28, 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 2))
