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
    #when bs = 1 BN = IN , we acutlly can use BN as default..
    def __init__(self,opt,input_channel,firstK=64,
                 n_downsample = 4,n_blocks = 9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        """
        n_block is the num of the res-block
        n_downsample is the num of the downsample layer
        firstK is the first kernel num of the conv layer
        """
        assert(n_blocks>=0)
        super(global_frame_generator, self).__init__()
        # self.generate_first_frame = generate_first_frame
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


    def forward(self,input,prev_frame,freature=None):
        """
        forward 1 big data will forward this many times.
        each "small" forward take the previous frames and (if use the last frames also) as the input
        pass though the init layer and the downsample layer
        then cat with the feature map from the Encoder (like a U-net but the feature are different)
        not sure should combine or not as quite like wasting caculation
        """
        x = input
        x = self.downsample(x)
        # inlcude resblock
        x = self.upsample(x)
        x = self.last(x)
        x = x + prev_frame
        return x

class Encoder(nn.Module):
    def __init__(self,input_channel,firstK=64):
        super(Encoder, self).__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ng = firstK

        layer_0 = [nn.Conv2d(input_channel, ng, kw, stride=2, padding=pw),nn.LeakyReLU(True)]
        # , nn.BatchNorm2d(ng)

        self.layer_0 = nn.Sequential(*layer_0)
        tempk = ng
        self.layer_1 = nn.Sequential(*[nn.Conv2d(ng * 1, ng * 2, kw, stride=2, padding=pw)])
        self.layer_2 = nn.Sequential(*[nn.Conv2d(ng * 2, ng * 4, kw, stride=2, padding=pw)])
        self.layer_3 = nn.Sequential(*[nn.Conv2d(ng * 4, ng * 8, kw, stride=2, padding=pw)])
        self.layer_4 = nn.Sequential(*[nn.Conv2d(ng * 8, ng * 8, kw, stride=2, padding=pw)])
        self.layer_5 = nn.Sequential(*[nn.Conv2d(ng * 8, ng * 8, kw, stride=2, padding=pw)])

        self.actvn = nn.LeakyReLU(0.2, False)
        self.fc_mu = nn.Sequential(*[nn.Linear(ng * 128, 256)])
        self.fc_var = nn.Sequential(*[nn.Linear(ng * 128, 256)])


    def forward(self,input):

        x = input

        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.layer_0(x)
        x = self.layer_1(self.actvn(x))
        x = self.layer_2(self.actvn(x))
        x = self.layer_3(self.actvn(x))
        x = self.layer_4(self.actvn(x))
        x = self.layer_5(self.actvn(x))
        # x = self.actvn(x)


        x = x.view(x.size(0), -1)
        # reshape
        # pass to 2 fc respectively
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

class pix2pix_generator(nn.Module):
    def __init__(self,input_channel ,K = 64 ,downsample_num = 6):
        super(pix2pix_generator, self).__init__()

        down = []
        up = []

        self.downstack = []
        self.upstack = []

        temp_k = K

        # 3 -> 64
        down += [[nn.Conv2d(in_channels=input_channel,out_channels=temp_k,stride=2,kernel_size=4,padding=1),
                            nn.LeakyReLU(True)]]

        # 64 -> max 512
        for i in range(1,downsample_num):
            output_channel = min(512,temp_k*2)
            down += [network.cNsN_K(input_channels=temp_k,stride=2,N=4,k=output_channel,padding=1,norm=nn.BatchNorm2d, activation=nn.LeakyReLU(True))]
            temp_k = min(512,temp_k*2)


        for i in range(len(down)):
            setattr(self,'downstack'+str(i),nn.Sequential(*down[i]))
            self.downstack.append(getattr(self,'downstack'+str(i)))


        for i in range(1, downsample_num):
            if i ==1 and (2**(downsample_num-i)*(K/2)) >= 512:
                up += [[nn.ConvTranspose2d(in_channels=temp_k,out_channels=temp_k,stride=2,kernel_size=4,padding=1),nn.Dropout(0.5),nn.ReLU(True)]]
            elif (2**(downsample_num-i)*(K/2)) >= 512:
                up += [[nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=temp_k,stride=2,kernel_size=4,padding=1),nn.Dropout(0.5),nn.ReLU(True)]]
            else:
                up += [[nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=int(temp_k/2),stride=2,kernel_size=4,padding=1),nn.BatchNorm2d(int(temp_k/2)),nn.ReLU(True)]]
                temp_k = int(temp_k / 2)

        for i in range(len(up)):
            setattr(self,'upstack'+str(i),nn.Sequential(*up[i]))
            self.upstack.append(getattr(self,'upstack'+str(i)))

        last = [nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=3,stride=2,kernel_size=4,padding=1),nn.BatchNorm2d(3),nn.ReLU(True)]
        setattr(self,'last',nn.Sequential(*last))


    def forward(self,input):
        skips = []
        x = input

        for down in self.downstack:
            # pass x to each downsampling , then push it in to skips

            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        # reversed the skips, which mean the first one goes to the last one
        for up,skip in zip(self.upstack,skips):

            x = up(x)
            x = torch.cat((x,skip),1)

        x = self.last(x)
        return x





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
    #when bs = 1 BN = IN , we acutlly can use BN as default..
    def __init__(self,opt,input_channel,firstK=64,
                 n_downsample = 4,n_blocks = 9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        """
        n_block is the num of the res-block
        n_downsample is the num of the downsample layer
        firstK is the first kernel num of the conv layer
        """
        assert(n_blocks>=0)
        super(global_frame_generator, self).__init__()
        # self.generate_first_frame = generate_first_frame
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


    def forward(self,input,prev_frame,freature=None):
        """
        forward 1 big data will forward this many times.
        each "small" forward take the previous frames and (if use the last frames also) as the input
        pass though the init layer and the downsample layer
        then cat with the feature map from the Encoder (like a U-net but the feature are different)
        not sure should combine or not as quite like wasting caculation
        """
        x = input
        x = self.downsample(x)
        # inlcude resblock
        x = self.upsample(x)
        x = self.last(x)
        x = x + prev_frame
        return x

class Encoder(nn.Module):
    def __init__(self,input_channel,firstK=64):
        super(Encoder, self).__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ng = 16
        layer_0 = [nn.Conv2d(input_channel, ng, kw, stride=2, padding=pw)]
        # , nn.BatchNorm2d(ng)
        self.layer_0 = nn.Sequential(*layer_0)
        tempk = ng
        self.layer_1 = nn.Sequential(*[nn.Conv2d(ng * 1, ng * 1, kw, stride=1, padding=pw),nn.Conv2d(ng * 1, ng * 1, kw, stride=2, padding=pw)])
        self.layer_2 = nn.Sequential(*[nn.Conv2d(ng * 1, ng * 1, kw, stride=1, padding=pw), nn.Conv2d(ng * 1, ng * 1, kw, stride=2, padding=pw)])
        self.layer_3 = nn.Sequential(*[nn.Conv2d(ng * 1, ng * 2, kw, stride=1, padding=pw), nn.Conv2d(ng * 2, ng * 2, kw, stride=2, padding=pw)])
        self.layer_4 = nn.Sequential(*[nn.Conv2d(ng * 2, ng * 4, kw, stride=1, padding=pw), nn.Conv2d(ng * 4, ng * 4, kw, stride=2, padding=pw)])

        self.actvn = nn.LeakyReLU(0.2, False)
        self.fc_mu = nn.Sequential(*[nn.Linear(4096, 256)])
        self.fc_var = nn.Sequential(*[nn.Linear(4096, 256)])


    def forward(self,input):

        x = input

        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer_0(x)

        x = self.layer_1(self.actvn(x))

        x = self.layer_2(self.actvn(x))

        x = self.layer_3(self.actvn(x))

        x = self.layer_4(self.actvn(x))

        # x = self.layer_5(self.actvn(x))
        # x = self.actvn(x)
        x = x.view(x.size(0), -1)

        # reshape
        # pass to 2 fc respectively
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

class pix2pix_generator(nn.Module):
    def __init__(self,input_channel ,K = 64 ,downsample_num = 6):
        super(pix2pix_generator, self).__init__()
        down = []
        up = []

        self.downstack = []
        self.upstack = []

        temp_k = K

        # 3 -> 64
        down += [[nn.Conv2d(in_channels=input_channel,out_channels=temp_k,stride=2,kernel_size=4,padding=1),
                            nn.LeakyReLU(True)]]

        # 64 -> max 512
        for i in range(1,downsample_num):
            output_channel = min(512,temp_k*2)
            down += [network.cNsN_K(input_channels=temp_k,stride=2,N=4,k=output_channel,padding=1,norm=nn.BatchNorm2d, activation=nn.LeakyReLU(True))]
            temp_k = min(512,temp_k*2)

        for i in range(len(down)):
            setattr(self,'downstack'+str(i),nn.Sequential(*down[i]))
            self.downstack.append(getattr(self,'downstack'+str(i)))


        for i in range(1, downsample_num):
            if i ==1 and (2**(downsample_num-i)*(K/2)) >= 512:
                up += [[nn.ConvTranspose2d(in_channels=temp_k,out_channels=temp_k,stride=2,kernel_size=4,padding=1),nn.Dropout(0.5),nn.ReLU(True)]]
            elif (2**(downsample_num-i)*(K/2)) >= 512:
                up += [[nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=temp_k,stride=2,kernel_size=4,padding=1),nn.Dropout(0.5),nn.ReLU(True)]]
            else:
                up += [[nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=int(temp_k/2),stride=2,kernel_size=4,padding=1),nn.BatchNorm2d(int(temp_k/2)),nn.ReLU(True)]]
                temp_k = int(temp_k / 2)

        for i in range(len(up)):
            setattr(self,'upstack'+str(i),nn.Sequential(*up[i]))
            self.upstack.append(getattr(self,'upstack'+str(i)))

        last = [nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=3,stride=2,kernel_size=4,padding=1),nn.BatchNorm2d(3),nn.ReLU(True)]
        setattr(self,'last',nn.Sequential(*last))

    def forward(self,input):
        skips = []
        x = input

        for down in self.downstack:
            # pass x to each downsampling , then push it in to skips

            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        # reversed the skips, which mean the first one goes to the last one
        for up,skip in zip(self.upstack,skips):

            x = up(x)
            x = torch.cat((x,skip),1)

        x = self.last(x)
        return x


class Deocder(nn.Module):
    def __init__(self,opt):
        super(Deocder, self).__init__()
        self.opt = opt
        self.sw = opt.input_size // (2**opt.n_downsample_global)
        self.sh = self.sw
        self.fc = nn.Sequential(*[nn.Linear(opt.z_dim, 8 * opt.firstK * self.sw * self.sh),nn.LeakyReLU(True)])
        # self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        # norm_layer = get_nonspade_norm_layer(opt, opt.norm_type)
        # cat_CH = opt.input_chan * 2 + 1
        # cat_CH = 0
        self.head = nn.Sequential(*[nn.ConvTranspose2d(in_channels=8 * opt.firstK * 2,
                                               out_channels=4 * opt.firstK,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1)])
        # 512 + 512 -> 256
        # 256 + 256 -> 128
        # 128 + 128 -> 64
        # 64 + 64 -> 32
        # 32 -> 3
        self.convtan1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=4 * opt.firstK * 2,
                                               out_channels=2 * opt.firstK,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1)])
        self.convtan2 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=2 * opt.firstK * 2,
                                                           out_channels= opt.firstK,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1,
                                                           output_padding=1)])
        self.convtan3 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=opt.firstK * 2,
                                                           out_channels=opt.firstK,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1,
                                                           output_padding=1)])

        self.last = nn.Sequential(*[nn.ConvTranspose2d(in_channels=opt.firstK,
                                              out_channels=opt.output_channel,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),nn.BatchNorm2d(opt.output_channel)])
        # self.padding = nn.ZeroPad2d(2)
        self.actvn = nn.LeakyReLU(0.2, False)
        # this is for extract the feature then cat with the above trans-conv
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.firstK
        norm_layer = get_nonspade_norm_layer(opt,opt.norm_type)
        self.layer1 = norm_layer(nn.Conv2d(opt.input_chan * 2 + 1, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        # this is for extract the feature then cat with the above transconv

        # the wrap the image just like vid2vid
        self.weight_up = nn.Sequential(*[nn.ConvTranspose2d(in_channels=8 * opt.firstK,
                                               out_channels=4 * opt.firstK,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),nn.ReLU(True),
                                         nn.ConvTranspose2d(in_channels=4 * opt.firstK,
                                               out_channels=2 * opt.firstK,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),nn.ReLU(True),
                                         nn.ConvTranspose2d(in_channels=2 * opt.firstK,
                                                           out_channels= opt.firstK,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1,
                                                           output_padding=1),nn.ReLU(True),
                                         # this is the final weight
                                         nn.Conv2d(in_channels=opt.firstK,
                                              out_channels=1, # we only need a number
                                              kernel_size=7,
                                              stride=1,
                                              padding=0),nn.Sigmoid()])

    def forward(self,previous_frame, input=None, cat_frame=None):
        if input is None:
            input = torch.randn(cat_frame.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=cat_frame.get_device())
        skips = []
        skip = self.actvn(self.layer1(cat_frame))
        skips.append(skip)
        skip = self.actvn(self.layer2(skips[-1]))
        skips.append(skip)
        skip = self.actvn(self.layer3(skips[-1]))
        skips.append(skip)
        skip = self.actvn(self.layer4(skips[-1]))
        skips.append(skip)
        skips.reverse()
        WEIGHT = self.weight_up(skips[0])
        WEIGHT_ = WEIGHT.expand_as(previous_frame)
        x = self.fc(input)
        x = x.view(-1, 8 * self.opt.firstK, self.sw , self.sh)
        x = self.actvn(x)
        x = torch.cat((x, skips.pop(0)), dim=1)
        x = self.head(x)
        x = self.actvn(x)
        x = torch.cat((x, skips.pop(0)), dim=1)
        x = self.convtan1(x)
        x = self.actvn(x)
        x = torch.cat((x,skips.pop(0)),dim=1)
        x = self.convtan2(x)
        x = self.actvn(x)
        x = torch.cat((x, skips.pop(0)), dim=1)
        x = self.convtan3(x)
        x = self.actvn(x)
        x = self.last(x)
        x = self.actvn(x)
        fake_next_frame = previous_frame * WEIGHT_ + x * (1 - WEIGHT_)
        return fake_next_frame


def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):

        if norm_type == 'none' or len(norm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % norm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

class ConvEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.firstK
        norm_layer = get_nonspade_norm_layer(opt,opt.norm_type)
        self.layer1 = norm_layer(nn.Conv2d(opt.input_chan * 4 + 1, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.input_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.input_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar