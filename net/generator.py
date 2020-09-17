import net
import net.network as network
import net.resnet as resnet
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
import copy

class global_frame_generator(nn.Module):

    def __init__(self,opt,input_channel,firstK=64,
                 n_downsample = 4,n_blocks = 9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect',Debug = False):
        """
        n_block is the num of the res-block
        n_downsample is the num of the downsample layer
        firstK is the first kernel num of the conv layer
        """
        assert(n_blocks>=0)
        super(global_frame_generator, self).__init__()
        self.Debug = Debug

        ngf = firstK

        downsample = []
        downsample += network.c7s1_k(input_channel,ngf)
        # 7 * 7 output 64 CH
        # downsample layer
        temp_K = ngf

        for i in range(1,n_downsample+1):
            ds = network.dk(temp_K,temp_K*2)
            temp_K*=2
            downsample+=ds

        # downsample layer
        # res-block
        for i in range(1,n_blocks+1):
            res = [resnet.ResK(dim=temp_K,padding_type=padding_type,norm_layer=norm_layer)]
            downsample += res
        # res-block
        # upsample layer
        # should cat with the output from the encoder if the encoder using a vector Z
        self.downsample = nn.Sequential(*downsample)
        upsample = []
        for i in range(1,n_downsample+1):
            us = network.uk(input_channels=temp_K,stride=2,N=3,k=int(temp_K/2))
            temp_K = int(temp_K/2)
            upsample += us
        self.upsample = nn.Sequential(*upsample)
        # upsample layer

        last = network.c7s1_k(temp_K,opt.output_channel)
        self.last = nn.Sequential(*last)


    def forward(self,input,freature=None):
        """
        forward 1 big data will forward this many times.
        each "small" forward take the previous frames and (if use the last frames also) as the input
        pass though the init layer and the downsample layer
        then cat with the feature map from the Encoder (like a U-net but the feature are different)
        not sure should combine or not as quite like wasting caculation
        """
        x = input
        x = self.downsample(x)
        x = self.upsample(x)
        x = self.last(x)
        return x

class Encoder(nn.Module):
    """
    with N conv + 2 liner fc for encode the feature of the difference of the current frame and the target frame

    """
    def __init__(self,input_channel,firstK=64):
        super(Encoder, self).__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ng = firstK

        layer_0 = [nn.Conv2d(input_channel, ng, kw, stride=2, padding=pw),nn.BatchNorm2d(ng,affine=True)]

        self.layer_0 = nn.Sequential(*layer_0)
        tempk = ng
        self.layer_1 = nn.Sequential(*[nn.Conv2d(ng * 1, ng * 2, kw, stride=2, padding=pw)])
        self.layer_2 = nn.Sequential(*[nn.Conv2d(ng * 2, ng * 4, kw, stride=2, padding=pw)])
        self.layer_3 = nn.Sequential(*[nn.Conv2d(ng * 4, ng * 8, kw, stride=2, padding=pw)])
        self.layer_4 = nn.Sequential(*[nn.Conv2d(ng * 8, ng * 8, kw, stride=2, padding=pw)])
        self.layer_5 = nn.Sequential(*[nn.Conv2d(ng * 8, ng * 8, kw, stride=2, padding=pw)])

        self.actvn = nn.LeakyReLU(0.2, False)
        self.fc_mu = nn.Linear(ng * 128, 256)
        self.fc_var = nn.Linear(ng * 128, 256)


    def forward(self,input):

        x = input

        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
            # not understand why the cropsize much bigger than these one, but still need a interpolate?
        print(x.shape)
        x = self.layer_0(x)
        x = self.layer_1(self.actvn(x))
        x = self.layer_2(self.actvn(x))
        x = self.layer_3(self.actvn(x))
        x = self.layer_4(self.actvn(x))
        x = self.layer_5(self.actvn(x))
        x = self.actvn(x)


        x = x.view(x.size(0), -1)
        # reshape
        # pass to 2 fc respectively
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar






