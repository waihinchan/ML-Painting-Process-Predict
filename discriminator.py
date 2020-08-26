import network
import torch
import torch.nn as nn
import numpy as np


class MultiscaleDiscriminator(nn.Module):
    def __init__(self,input_channel,k=64,n_layers=3,
                 norm_layer = nn.BatchNorm2d,sigmoid=False,
                 num_D=3,getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NlayerDiscriminator(input_channel,k,n_layers,norm_layer,sigmoid,getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'dis_block'+str(j)))
                    # scaleN + layerN = that netD's dis_block
            else:
                setattr(self,'scale'+str(i),netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self,input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            # num_D-1-i -> start from the last
            if self.getIntermFeat:
                # iter each_scale' dis block
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))

            result.append(self.singleD_forward(model, input_downsampled))
            # get the result(if getinterm, each single D would have multi result, then would be numD * numlayers result)
            if i != (num_D-1):
                # downsample each result except for the first one
                input_downsampled = self.downsample(input_downsampled)
        return result

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

class NlayerDiscriminator(nn.Module):
    def __init__(self,input_channel,k=64,n_layers = 3,
                 norm_layer = nn.BatchNorm2d,
                 sigmoid = False,getIntermFeat=False):
        super(NlayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kernal_w = 4
        padding_w = int(np.ceil((kernal_w-1.0)/2))
        sequence = [[nn.Conv2d(input_channel, k, kernel_size=kernal_w, stride=2, padding=padding_w),
                     nn.LeakyReLU(0.2, True)]]
        temp_k = k
        for i in range(1,n_layers):
            # remain 1 last with stride 1
            temp_k = min(256,temp_k)
            sequence += [network.cNsN_K(temp_k,2,N=kernal_w,k=temp_k*2,padding=padding_w,norm=norm_layer,stride=2,activation=nn.LeakyReLU(0.2,True))]
            # CNsN_K return [model]
            # add one more [] outside
            temp_k *= 2

        sequence += [network.cNsN_K(temp_k,2,N=kernal_w,k=temp_k*2,padding=padding_w,norm=norm_layer,stride=1,activation=nn.LeakyReLU(0.2,True))]
        # the last is with stride = 1 conv layer + norm + activation
        sequence += [[nn.Conv2d(temp_k*2, 1, kernel_size=kernal_w, stride=1, padding=padding_w)]]
        # output the last (patch)
        if sigmoid:
            sequence += [[nn.Sigmoid()]]
        if getIntermFeat:
            for n in range(len(sequence)):
                # iter each [] of the sequence
                # separate each conV
                setattr(self, 'dis_block'+str(n), nn.Sequential(*sequence[n]))
                # the last would be the sigmoid, not sure need combine to the last Conv or not.
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
                # sequence[n] = [model]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self,input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                # head + n_layers + last + not include(sigmoid(or not use))
                model = getattr(self, 'dis_block'+str(n))
                res.append(model(res[-1]))
                return res[1:]
            # get each block's result, retrun except for the first one
        else:
            return self.model(input)




# ******************************************************************************** #
"""
this is reference to the pix2pix patchGAN, some details are different.
didn't use BatchNorm (which use in pix2pixHD)
use instanceNorm
didn't use get_Interm_Feat as for temporary didn't need
"""

class multiProgressDiscriminator(nn.Module):
    def __init__(self):
        super(multiProgressDiscriminator, self).__init__()

class patchGAN(nn.Module):
    def __init__(self,input_channel, K=64, n_layers=4):
        super(patchGAN, self).__init__()
        model = []
        model += [nn.Conv2d(input_channel, K, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        temp_k = K

        for i in range(1, n_layers):
            # the first one is at the top
            # not more than 512 filters
            if temp_k <= 256:
                model += network.cNsN_K(temp_k,2,N=4,k=temp_k*2,padding=1,norm=nn.InstanceNorm2d,activation=nn.LeakyReLU(0.2,True))
                temp_k *= 2
            else:
                model += network.cNsN_K(temp_k,2,N=4,k=temp_k,padding=1,norm=nn.InstanceNorm2d,activation=nn.LeakyReLU(0.2,True))

        model += network.cNsN_K(temp_k,2,N=4,k=1,padding=1,norm=nn.InstanceNorm2d,activation=nn.Sigmoid())

        # sequence_stream = []
        #
        # for i in range(len(model)):
        #     sequence_stream += [model[i]]

        self.model = nn.Sequential(*model)


    def forward(self,input):
        return self.model(input)

# ******************************************************************************** #




# some test

# input = torch.rand(1,3,1024,1024)
# ma = patchGAN(3)
# print(ma(input).shape)
# print(ma)

# some test
