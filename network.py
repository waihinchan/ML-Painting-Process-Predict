import torch
from mydataprocess import dataset
import torch.nn as nn
import numpy as np
import generator
import discriminator

def create_G(input_channel ,K = 64 ,downsample_num = 6):
    netG = generator.pix2pix_generator(input_channel,K = K, downsample_num = downsample_num,)
    netG.apply(init_weights)
    return netG



def create_D(input_channel,K = 64,n_layers = 4):
    netD = discriminator.patchGAN(input_channel,K = K,n_layers=n_layers)
    netD.apply(init_weights)
    return netD


# some tools

def init_weights(l):
    classname = l.__class__.__name__
    if classname.find('Conv') != -1:
        l.weight.data.normal_(0.0, 0.02)
    if classname.find('BatchNorm2d') != -1:
        l.weight.data.normal_(1.0, 0.02)
        l.bias.data.fill_(0)



def get_norm_layer(norm_type='instance'):
    # base on the params define a norm_layer is a batchnorm or a instance norm
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        # 这是一个函数的包装器
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# some tools



# basic block

def cNsN_K(input_channels,stride,N,k,padding,norm,activation):
    """
    :param input_channels:
    :param stride:
    :param N: kernel size
    :param k: filters
    :param norm: instance or batch
    :param activation: relu or leaky relu
    :return:
    """
    model = [nn.Conv2d(input_channels,k,kernel_size=N,stride=stride,padding=padding),
             norm(k),
             activation
             ]
    return model

def c7s1_k(input_nc,k):
    return cNsN_K(input_channels=input_nc,stride=1,N=7,k=k,padding=0,norm=nn.InstanceNorm2d,activation=nn.ReLU(True))

def dk(input_nc,k):
    """
    downSampling with
    3*3 Convolution-InstanceNorm-ReLU layer with k filters, and stride 2.
    :param input_nc: input channels
    :param k: output. k filters
    :param padding: padding
    :return: a list with the above structure
    """
    return cNsN_K(input_channels=input_nc,stride=2,N=3,k=k,padding=1,norm=nn.InstanceNorm2d,activation=nn.ReLU(True))

def uk(input_channels,stride,N,k,norm = nn.InstanceNorm2d, activation = nn.ReLU(True)):
    """
    a 3*3 fractional-strided-Convolution- InstanceNorm-ReLU layer with k filters, and stride 1/2 .
    maybe could use stride = 1/2 in cnsn-k as the tranConv is = 1/2 stride Conv
    not sure the difference
    :param input_channels:
    :param stride:
    :param N: kernel size
    :param k: filters
    :param norm: instance or batch
    :param activation: relu or leaky relu
    :return: list with the above structure
    """
    model = [nn.ConvTranspose2d(input_channels,k,kernel_size=N,stride=stride,padding=1,output_padding=1),
             norm(k),
             activation]
    return model


def ck(input_nc,k):
    """
    4*4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2.
    input_channels,stride,N,k,padding,norm,activation
    :param input_nc:
    :param k:
    :param padding:
    :return:
    """
    return cNsN_K(input_channels=input_nc,k=k,N=4,padding=1,norm=nn.InstanceNorm2d,activation=nn.LeakyReLU(0.2,True),stride=2)








# ck(10,10)
# c7s1_k(10,10)
# uk(3,k=10,stride=1,N=3,norm=nn.InstanceNorm2d,activation=nn.ReLU)
# dk(10,50)
# print(ResK(10,'zero',norm_layer=nn.InstanceNorm2d).conv_block)



