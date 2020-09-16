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

    def __init__(self,input_channel,firstK=64,
                 n_downsample = 4,n_blocks = 9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks>=0)
        super(global_frame_generator, self).__init__()
        # the input will be the cat of the xT and the xt-1
        ngf = firstK
        global_network = []
        global_network += network.c7s1_k(input_channel,ngf)
        temp_K = ngf
        for i in range(1,n_downsample+1):
            ds = network.dk(temp_K,temp_K*2)
            temp_K*=2
            global_network+=ds
        for i in range(1,n_blocks+1):
            res = [resnet.ResK(dim=temp_K,padding_type=padding_type,norm_layer=nn.InstanceNorm2d)]
            global_network += res

        for i in range(1,n_downsample+1):
            us = network.uk(input_channels=temp_K,stride=2,N=3,k=int(temp_K/2))
            temp_K = int(temp_K/2)
            global_network += us

        global_network += network.c7s1_k(temp_K,input_channel)

        global_network = nn.Sequential(*global_network)
        self.model = global_network

    def forward(self,input):
        # w.r.t generate 1 frame
        # forward 1 big data = genereate 1st frame then pass into it again then pass into it again again...
        return self.model(input)

class Encoder(nn.Module):
    """
    with N conv + 2 liner fc for encode the feature of the difference of the current frame and the target frame

    """
    def __init__(self,input_channel,firstK):
        super(Encoder, self).__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ng = firstK
        # the input will be the prev_fram(or blank image) cat with the real_last_frame
        # for encode
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
        x = self.layer_0(x)
        x = self.layer_1(self.actvn(x))
        x = self.layer_2(self.actvn(x))
        x = self.layer_3(self.actvn(x))
        x = self.layer_4(self.actvn(x))
        x = self.layer_5(self.actvn(x))
        x = self.actvn(x)
        print(x.shape)
        x = x.view(x.size(0), -1)

        # reshape
        # pass to 2 fc respectively
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar






