import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
"""
remain to sort:
1.a Normalization need to be added
2.effect is not good enough, but can be a part of the loss function
3.remain to move this part to the Loss page
4.some problems about the resolution
"""

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    # make a fake batch size
    return image.to(device,torch.float)

unloader = transforms.ToPILImage()  # reconvert into PIL image
def imshow(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.figure()
    plt.imshow(image)
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use GPU or CPU

class ContentLoss(nn.Module):
    def __init__(self,target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        # here we put the content image(the input image)
    def __call__(self, input):
        self.loss = F.mse_loss(input,self.target)
        return input
    # calculate per pixels difference

from Loss import gram


class StyleLoss(nn.Module):
    def __init__(self,target):
        super(StyleLoss, self).__init__()
        garm_target = gram(target)
        self.target = garm_target.detach()
        # here is the style image we want to transfer to

    def forward(self,input):
        gram_input = gram(input)
        self.loss = F.mse_loss(gram_input,self.target)
        return input
    # calculate the mse different between the input and target


class Normalization(nn.Module):
    def __init__(self,mean,std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)
    def forward(self,img):
        return (img - self.maen)/self.std




imgsize = 512 if torch.cuda.is_available() else 128
# use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imgsize),
    transforms.ToTensor()
])
# pre-process pipeline


style_img = image_loader("./dataset/070/011-d.png")
content_img = image_loader("./dataset/070/018-a.png")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

# imshow(style_img)


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# input_img = torch.randn(content_img.data.size(), device=device)
# optimizer = optim.LBFGS([input_img.requires_grad_()])
optimizer = optim.LBFGS([content_img.requires_grad_()])


from Loss import Vgg19
model = Vgg19()
print(model)

target_styles = model(style_img)
# for i in target_styles:
#     imshow(i)
now = [0]
total = 300
net = []

for sl in target_styles:
    net.append(StyleLoss(sl))

contents = model(content_img)
cl = ContentLoss(contents[3])

def run():
    while now[0] < total:
        def closure():
            # input_img.data.clamp_(0, 1)
            content_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            loss = 0
            input_styles = model(content_img)
            for il,sl in zip(input_styles,net):
                sl(il)
                loss+=sl.loss

            loss*=1000000
            cl(input_styles[3])
            loss+=cl.loss
            loss.backward()
            print(loss)
            now[0]+=1
            return loss

        optimizer.step(closure)

# input_img.data.clamp_(0, 1)
content_img.data.clamp_(0, 1)
# run()
# imshow(content_img)