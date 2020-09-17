import torch
from mydataprocess import dataset
import torch.nn as nn
import numpy as np
import torch.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

# some tools

def init_weights(l):
    classname = l.__class__.__name__
    if classname.find('Conv') != -1:
        l.weight.data.normal_(0.0, 0.02)
    if classname.find('BatchNorm2d') != -1:
        l.weight.data.normal_(1.0, 0.02)
        l.bias.data.fill_(0)

# this is for the light flow
def get_grid(batchsize, rows, cols, gpu_id=0, dtype=torch.float32):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False

    if dtype == torch.float16: t_grid = t_grid.half()
    return t_grid.cuda(gpu_id)
# this is for the light flow

# this is for spade network for generate the spectral norm layer
def add_norm_layer(layer):
    # we definely use spectral
    layer = spectral_norm(layer)
    # then we use batch norm
    if getattr(layer, 'bias', None) is not None:
        delattr(layer, 'bias')
    layer.register_parameter('bias', None)
    # remove bias in the previous layer, which is meaningless
    # since it has no effect after normalization
    # not sure where the theory from, just do it!
    out_channels = getattr(layer, 'out_channels') if  hasattr(layer, 'out_channels') else layer.weight.size(0)
    norm_layer = nn.BatchNorm2d(out_channels,affine=True)
    # get the previous layer's input size, then add a batchNorm(we use BN rather than IN, not provide any option)
    return nn.Sequential(layer, norm_layer)
# this is for spade network




# some tools



# basic structure

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
    c7s1k = []
    c7s1k+=[nn.ReflectionPad2d(3)]
    c7s1k+=cNsN_K(input_channels=input_nc,stride=1,N=7,k=k,padding=0,norm=nn.InstanceNorm2d,activation=nn.ReLU(True))
    return c7s1k

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



# basic structure
class SpadeBN(nn.Module):
    def __init__(self, nf,norm_nc):
        super(SpadeBN, self).__init__()
        nhidden = 128
        self.bn = nn.BatchNorm2d(nf, affine=False)
        # the params will determind by seg map
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(nf, nhidden, kernel_size=1, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, input, segmap):
        size = input.size()[-2:]
        #not sure the size would be H * W
        #(BS,CH,H,W)
        #not sure what mask is , maybe is the segmap tensor
        segmap = F.interpolate(segmap.float(), size=size,mode='nearest')
        interim_conv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(interim_conv)
        beta = self.mlp_beta(interim_conv)
        return self.bn(input) * (gamma+1) + beta






