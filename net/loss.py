import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self,device,lsgan=True):
        super(GANLoss, self).__init__()
        self.device = device
        if lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def transofrm_tensor(self,input,label):
        if label:
            target_tensor = torch.ones(input.shape)

        else:
            target_tensor = torch.zeros(input.shape)

        return target_tensor.to(self.device)

    def __call__(self, input, label):
        # remain to be test
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.transofrm_tensor(pred, label)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.transofrm_tensor(input[-1], label)
            return self.loss(input[-1], target_tensor)
        # remain to be test

from torchvision import models

class Vgg19(torch.nn.Module):
    # use pre-trained net work for the perceptual loss
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        # should be conv block (not sure, just quick check the source code)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        # 5 feature/style extract slice
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class pixHDversion_perceptual_loss(nn.Module):
    def __init__(self, gpu_ids,loss = nn.MSELoss()):
        super(pixHDversion_perceptual_loss, self).__init__()
        self.vgg = Vgg19()
        if gpu_ids > 0:
            assert(torch.cuda.is_available())
            self.vgg.cuda()
        self.criterion = loss

        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        # should be something divided by the element number or something average

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # target image and input image -> return a [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        loss = 0
        for i in range(len(x_vgg)):
            # add each slice output * weights
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            # y_vgg no require grad
        return loss

def gram(input):
    (bs, ch, h, w) = input.size()
    features = input.view(bs * ch, w * h)
    Gram = torch.mm(features,features.t())
    # print(Gram)
    Gram = Gram.div(bs*ch*h*w)
    return Gram


# y=A(x), z=B(y) 求B中参数的梯度，不求A中参数的梯度
# y = B(A(x))
# 第一种方法
# y = A(x)
# z = B(y.detach())
# z.backward()
# wo = Vgg19()
# print(wo)


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):

        batch_size = x.size()[0]
        h_x = x.size()[2]

        w_x = x.size()[3]

        count_h = (x.size()[2]-1) * x.size()[3]

        count_w = x.size()[2] * (x.size()[3] - 1)

        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # vertical pixel - vertical pixel abs add together
        # from the second pixel start - from 0 to the last two
        # No.2 - No.1023
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        # the same
        # 就是用 相邻像素想减之后的平方和
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
        # why *2

# KL Divergence loss used in VAE with an image encoder
# get from spade
class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
# get from spade