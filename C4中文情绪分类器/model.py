import torch.nn as nn
from preprocessing import train_label, train_data, valida_label, valid_data, test_data, test_label, diction
import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from C6Bee_or_Ant.log import get_logger
from sklearn.metrics import classification_report
import logging

model = nn.Sequential(nn.Linear(len(diction), 10), nn.ReLU(), nn.Linear(10, 2), nn.LogSoftmax(dim=1), )


def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

print(train_data)
cost = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
records = []
logger = get_logger('log/sentiment classification.log')
logger.info('start training!')
losses = []
for epoch in range(10):
    for i, data in enumerate(tqdm(zip(train_data, train_label))):
        x, y = data
        x = Variable(torch.FloatTensor(x).view(1, -1))
        y = Variable(torch.LongTensor(np.array(([y]))))
        optimizer.zero_grad()
        predict = model(x)
        pred = predict.argmax(dim=1)
        loss = cost(predict, y)
        losses.append(loss.data.detach().numpy())
        loss.backward()
        optimizer.step()

        if i % 3000 == 0:
            val_loss = []
            rights = []
            for j, val in enumerate(zip(valid_data, valida_label)):
                x, y = val
                x = Variable(torch.FloatTensor(x).view(1, -1))
                y = Variable(torch.LongTensor(np.array(([y]))))
                predict = model(x)
                right = rightness(predict, y)
                rights.append(right)
                loss = cost(predict, y)
                val_loss.append(loss.detach())

            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            logging.info(
                'Training epoch：{} , Loss: {:.6f}\t, Validation Loss：{:.2f}%\t, Validation Accuracy: {:.2f}%'.format(
                    epoch, np.mean(losses), np.mean(val_loss), right_ratio))
            records.append([np.mean(losses), np.mean(val_loss), right_ratio])

vals = []
for data, target in zip(test_data, test_label):
    x = Variable(torch.FloatTensor(data).view(1, -1))
    y = Variable(torch.LongTensor(np.array(([target]))))
    output = model(x)
    val = rightness(output, y)
    vals.append(val)
rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 1.0 *rights[0]/rights[1]
print(right_rate)