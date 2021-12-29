import torch
from torch.autograd import Variable
import config
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

img_trans = transforms.Compose([transforms.Resize((config.imsize, config.imsize)), transforms.ToTensor()])


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(img_trans(image))
    image = image.unsqueeze(0)
    return image


style_img = image_loader(config.style).type(dtype)
content_img = image_loader(config.content).type(dtype)

assert style_img.size() == content_img.size()


def imshow(tensor, title=None):
    image = tensor.clone().cpu()
    image = image.view(3, config.imsize, config.imsize)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


unloader = transforms.ToPILImage()
plt.ion()

plt.figure()
imshow(style_img.data, title='Style Image')

plt.figure()
imshow(content_img.data, title='Content Image')
