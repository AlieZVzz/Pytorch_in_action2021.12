from config import *
from VGG import *
from utils import *
from tqdm import tqdm

input_img = Variable(torch.randn(content_img.data.size())).type(dtype)
if use_cuda:
    input_img = input_img.cuda()
    content_img = content_img.cuda()
    style_img = style_img.cuda()

plt.figure()
imshow(input_img.data, title='Input_image')

input_param = nn.Parameter(input_img.data)
optimizer = torch.optim.LBFGS([input_param])


print('The migration style model is being constructed')
for i in tqdm(range(config.num_steps)):
    input_param.data.clamp_(0, 1)
    optimizer.zero_grad()
    model(input_param)
    style_score = 0
    content_score = 0
    for sl in style_losses:
        style_score += sl.backward()
    for cl in content_losses:
        content_score += cl.backward()
    if i % 50 == 0:
        print("Runing {} steps: ".format(i))
        print("style_loss: {:.4f}, content_loss: {:.4f}".format(style_score, content_score))
        print()


    def closure():
        return style_score + content_score
    optimizer.step(closure)

output = input_param.data.clamp_(0, 1)
plt.figure()
imshow(output, title="Output Image")

plt.ioff()
plt.show()