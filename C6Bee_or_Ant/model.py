import logging
import torch.optim
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from data import train_loader, val_loader, rightness
from log import get_logger
from sklearn.metrics import classification_report

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

use_cuda = torch.cuda.is_available()
print(use_cuda)

model = model.cuda()
record = []
num_epochs = 20
model.train(True)

logger = get_logger('log/Bee_or_Ant.log')
logger.info('start training!')
for epoch in range(num_epochs):
    train_rights = []
    train_losses = []
    train_preds = []
    train_trues = []
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = Variable(data), Variable(target)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = rightness(output, target)
        train_rights.append(right)
        loss = loss.cpu() if use_cuda else loss
        train_losses.append(loss.data.numpy())
        if batch_idx % 100 == 0:
            model.eval()
            val_rights = []
            for (data, target) in val_loader:
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                right = rightness(output, target)
                val_rights.append(right)
        pred = output.argmax(dim=1)
        train_preds.extend(pred.detach().cpu().numpy())
        train_trues.extend(target.detach().cpu().numpy())
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
        val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
        record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))

    logging.info(
        'Training epoch：{} [{}/{} ({:.0f}%)]\t, Loss: {:.6f}\t, Train Accuracy：{:.2f}%\t, Validation Accuracy: {:.2f}%'
            .format(epoch, batch_idx * len(data), len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(),
                    100. * train_r[0] / train_r[1],
                    100. * val_r[0] / val_r[1]))
    logger.info('\n' + classification_report(train_trues, train_preds))

logger.info('finish training!')
record = [(i.cpu(), j.cpu()) for i, j in record]

plt.figure(figsize=(10, 7))
plt.plot(record)
plt.xlabel('Steps')
plt.ylabel('Error rate')
plt.show()
