import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm



image_size = 28
num_classes = 10
num_epochs = 10
batch_size = 256

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
indices = range(len(test_dataset))
indices_val = indices[:5000]
indices_test = indices[5000:]

sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_val)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_test)

idx = 100
muteimg = train_dataset[idx][0].numpy()
plt.imshow(muteimg[0, ...])
plt.title(train_dataset[idx][1])

depth = [4, 8]


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, 5), padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(depth[0], depth[1], (5, 5), padding=2)
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=-1)
        return x

    def retrieve_feature(self, x):
        feature_map1 = torch.relu(self.conv1(x))
        x = self.pool1(feature_map1)
        feature_map2 = torch.relu(self.conv2(x))
        return feature_map1, feature_map2


model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

record = []
weights = []


def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)



for epoch in range(num_epochs):
    train_rights = []
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # data, target = Variable(data), Variable(target)
        model.train()

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = rightness(output, target)
        train_rights.append(right)
        # print(train_rights)

        if batch_idx % 100 == 0:
            model.eval()
            val_rights = []
            for (data, target) in validation_loader:
                # data, target = Variable(data), Variable(target)
                output = model(data)
                right = rightness(output, target)
                val_rights.append(right)
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
        val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
        record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))
        weights.append([model.conv1.weight.data.clone(), model.conv1.bias.data.clone(), model.conv2.weight.data.clone(),
                        model.conv2.bias.data.clone()])
    print('Training epoch：{} [{}/{} ({:.0f}%)]\t, Loss: {:.6f}\t, Train Accuracy：{:.2f}%\t, Validation Accuracy: {:.2f}'
          .format(epoch, batch_idx * len(data), len(train_loader),
                  100. * batch_idx / len(train_loader), loss.item(),
                  100. * train_r[0] / train_r[1],
                  100. * val_r[0] / val_r[1]))

model.eval()
vals = []

for data, target in test_loader:
    output = model(data)
    val = rightness(output, target)
    vals.append(val)

rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
accuracy = 1.0 * rights[0] / rights[1]
print(accuracy)
print(record)

plt.figure(figsize=(10, 7))
plt.plot(record)
plt.xlabel('Steps')
plt.ylabel('Error rate')
plt.show()
