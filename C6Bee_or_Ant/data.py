import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch.utils.data

datadir = 'hymenoptera_data'
image_size = 224

train_datasets = datasets.ImageFolder(os.path.join(datadir, 'train'),
                                      transform=transforms.Compose([transforms.RandomResizedCrop(image_size),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                                         [0.229, 0.224, 0.226])]))
val_datasets = datasets.ImageFolder(os.path.join(datadir, 'val'),
                                      transform=transforms.Compose([transforms.Scale(256),
                                                                    transforms.CenterCrop(image_size),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                                         [0.229, 0.224, 0.226])]))

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=4, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=4, shuffle=True, num_workers=0)

num_class = len(train_datasets.classes)

def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)