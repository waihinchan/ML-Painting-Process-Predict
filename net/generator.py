import net
import net.network as network
import net.resnet as resnet
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm

# 稳定版
class Deocder(nn.Module):
    def __init__(self,opt):
        super(Deocder, self).__init__()
        self.opt = opt
        self.sw = opt.input_size // (2**opt.n_downsample_global)
        self.sh = self.sw
        self.fc = nn.Sequential(*[nn.Linear(opt.z_dim, 8 * opt.firstK * self.sw * self.sh),nn.LeakyReLU(True)])
        # self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        # norm_layer = get_nonspade_norm_layer(opt, opt.norm_type)

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
        self.actvn = nn.LeakyReLU(0.2, False)
        # this is for extract the feature then cat with the above trans-conv
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.firstK
        norm_layer = get_nonspade_norm_layer(opt,opt.norm_type)

        cat_CH = opt.input_chan * 2 + opt.label_CH + 2 if self.opt.use_label else opt.input_chan * 2
        # last current +  one-hot + (edge + segmap)
        self.layer1 = norm_layer(nn.Conv2d(cat_CH, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
    # **********************************WEIGHT CONV LAYER **********************************
    def forward(self, cat_frame,input=None):
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
        img_feature = skip
        skips.append(skip)
        skips.reverse()
        x = self.fc(input)
        x = x.view(-1, 8 * self.opt.firstK, self.sw , self.sh)
        x = self.actvn(x) # TODO fc has the actvn... i forget
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

        return x

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
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.firstK
        norm_layer = get_nonspade_norm_layer(opt,opt.norm_type)
        input_CH = opt.input_chan * 4 + opt.label_CH + 2 if self.opt.use_label else opt.input_chan * 4
        # current difference next last + onehot(label_CH) + 1(segmap) + 1(edgemap)
        self.layer1 = norm_layer(nn.Conv2d(input_CH, ndf, kw, stride=2, padding=pw))
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
#  稳定版

#  ************************ experiment ****************************** #

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from net.resnet import ResK
import copy
class Encoder2(nn.Module):
    def __init__(self,opt):
        super(Encoder2, self).__init__()
        self.opt = opt
        inputCH = opt.input_chan*3 # current + last + next is fixed
        CH_list = 'current,last,next,'
        if opt.use_difference:
            inputCH += opt.input_chan # 12
            CH_list+='difference,'
        if opt.use_label:
            inputCH+=1 # 13
            CH_list+='segmap,'
        if opt.use_wireframe:
            inputCH += 1
            CH_list+='wireframe,'
        if opt.use_instance:
            inputCH += 1 # 14
            CH_list+='instance,'
        if opt.use_degree == 'wrt_position':
            inputCH += (opt.granularity+1)
            CH_list+='wrt_position,'
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ng = opt.firstK
        norm_layer = nn.InstanceNorm2d
        self.layer_0 = nn.Sequential(*[nn.Conv2d(inputCH, ng * 1, kw, stride=2, padding=pw),
                                       norm_layer(ng * 1)])
        self.layer_1 = nn.Sequential(*[nn.Conv2d(ng * 1, ng * 2, kw, stride=2, padding=pw),
                                       norm_layer(ng * 2)])
        self.layer_2 = nn.Sequential(*[nn.Conv2d(ng * 2, ng * 4, kw, stride=2, padding=pw),
                                       norm_layer(ng * 4)])
        self.layer_3 = nn.Sequential(*[nn.Conv2d(ng * 4, ng * 8, kw, stride=2, padding=pw),
                                       norm_layer(ng * 8)])
        self.layer_4 = nn.Sequential(*[nn.Conv2d(ng * 8, ng * 8, kw, stride=2, padding=pw),
                                       norm_layer(ng * 8)])
        if opt.input_size >= 256:
            self.layer_5 = nn.Sequential(*[nn.Conv2d(ng * 8, ng * 8, kw, stride=2, padding=pw),
                                           norm_layer(ng * 8)])
        self.actvn = nn.LeakyReLU(0.2, False)
        coefficient = 128 if self.opt.input_size>=256 else 512
        self.fc_mu = nn.Sequential(*[nn.Linear(ng * coefficient, opt.z_dim)])
        self.fc_var = nn.Sequential(*[nn.Linear(ng * coefficient, opt.z_dim)])
        print('Encoder is using:' + CH_list)
    def forward(self,input):
        x = input
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.actvn(self.layer_0(x))
        x = self.actvn(self.layer_1(x))
        x = self.actvn(self.layer_2(x))
        x = self.actvn(self.layer_3(x))
        x = self.actvn(self.layer_4(x))
        if self.opt.input_size >= 256:
            x = self.actvn(self.layer_5(x))
        # x = self.actvn(x) # don't know why need to pass to 2 actvn...
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar
# **************************************** Decoder *************************************** #

class Decoder2(nn.Module):
    def __init__(self,opt):
        super(Decoder2, self).__init__()
        self.opt = opt
        self.sw = opt.input_size // (2**opt.upsample_num_)
        self.sh = self.sw
        self.newng = int(opt.firstK) if opt.input_size >= 256 else int(opt.firstK / 2)
        self.fc = nn.Sequential(*[nn.Linear(opt.z_dim, 8 * self.newng * self.sw * self.sh),nn.LeakyReLU(True)])
        # ************************************************ the z to fc ************************************************
        # ************************************** the downsampler is the same like the encoder *************************
        inputCH = opt.input_chan * 2  # current + last  6
        CH_list = 'current,last,'
        if opt.use_label:
            inputCH += 1
            CH_list += 'segmap,'
        if opt.use_wireframe:
            inputCH += 1
            CH_list += 'wireframe,'
        if opt.use_instance:
            inputCH += 1
            CH_list += 'instance,'
        if opt.use_degree == 'wrt_position':
            inputCH += (opt.granularity+1)
            CH_list += 'wrt_position,'

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ng = opt.firstK
        norm_layer = nn.InstanceNorm2d
        self.layer_0 = nn.Sequential(*[nn.Conv2d(inputCH, ng * 1, kernel_size=7, stride=1),
                                       norm_layer(ng * 1)])
        self.layer_1 = nn.Sequential(*[nn.Conv2d(ng * 1, ng * 2, kw, stride=2, padding=pw),
                                       norm_layer(ng * 2)])
        self.layer_2 = nn.Sequential(*[nn.Conv2d(ng * 2, ng * 4, kw, stride=2, padding=pw),
                                       norm_layer(ng * 4)])


        self.layer_3 = nn.Sequential(*[nn.Conv2d(ng * 4, self.newng * 8, kw, stride=2, padding=pw),
                                       norm_layer(self.newng * 8)])
        self.layer_4 = nn.Sequential(*[nn.Conv2d(self.newng * 8, self.newng * 8, kw, stride=2, padding=pw),
                                       norm_layer(self.newng * 8)])
        if opt.input_size >= 256:
            self.layer_5 = nn.Sequential(*[nn.Conv2d(ng * 8, ng * 8, kw, stride=2, padding=pw),
                                           norm_layer(ng * 8)])

        self.actvn = nn.LeakyReLU(0.2, False)
        self.padding = nn.ReplicationPad2d(3)
        if opt.use_restnet:
            dim = self.newng*16
            for i in range(opt.n_blocks):
                setattr(self, 'ResnetBlock' + str(i), ResK(dim=dim,padding_type='reflect',norm_layer=norm_layer))


        self.up1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=16 * self.newng,
                                               out_channels=8 * self.newng,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),norm_layer(8 * self.newng)])
        self.up2 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=8 * self.newng,
                                               out_channels=4 * self.newng,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),norm_layer(4 * self.newng)])
        self.up3 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=4 * self.newng,
                                                           out_channels= 2 * self.newng,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1,
                                                           output_padding=1),norm_layer(2 * self.newng)])
        if self.opt.input_size >= 256:
            self.up4 = nn.Sequential(*[nn.ConvTranspose2d(in_channels= 2* self.newng,
                                                               out_channels=self.newng,
                                                               kernel_size=3,
                                                               stride=2,
                                                               padding=1,
                                                               output_padding=1),norm_layer(self.newng)])

        # this is fixed
        self.last = nn.Sequential(*[nn.ConvTranspose2d(in_channels=opt.firstK,
                                              out_channels=opt.output_channel,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1),norm_layer(opt.output_channel)])
        # ****************** up ************************************ #
        if not opt.use_raw_only:
            self.weight_up1 = copy.deepcopy(self.up1)
            self.weight_up2 = copy.deepcopy(self.up2)
            self.weight_up3 = copy.deepcopy(self.up3)
            if opt.input_size>=256:
                self.weight_up4 = copy.deepcopy(self.up4)
            self.weight_last = copy.deepcopy(self.last)
            self.weight_final = nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(opt.output_channel, opt.output_channel, kernel_size=7, padding=0), nn.Sigmoid()])
        # 如果用weight的话 其实思路和wrtposition很像的，所以意义不是很大？
        # 都试试
        print('Decoder is using:' + CH_list + 'as input')
    def forward(self, cat_frame,z=None):
        reshape_degree = None
        if z is None:
            z = torch.randn(cat_frame.size(0), self.opt.z_dim,dtype=torch.float32, device=cat_frame.get_device())

        x = self.fc(z)
        x = x.view(self.opt.batchSize,8 * self.newng, self.sw , self.sh)
        # downsampler
        features = self.padding(cat_frame)
        # print(features.shape)
        features = self.actvn(self.layer_0(features))
        # print(features.shape)
        features = self.actvn(self.layer_1(features))
        # print(features.shape)
        features = self.actvn(self.layer_2(features))
        # print(features.shape)
        features = self.actvn(self.layer_3(features))
        # print(features.shape)
        features = self.actvn(self.layer_4(features))
        # print(features.shape)
        if self.opt.input_size >= 256:
            features = self.actvn(self.layer_5(features))
        # the shape of the feature should be the same like the x

        x = torch.cat([x,features],dim=1)
        w = None
        if not self.opt.use_raw_only:
            w = self.actvn(self.weight_up1(x))
            w = self.actvn(self.weight_up2(w))
            w = self.actvn(self.weight_up3(w))
            if self.opt.input_size>=256:
                w = self.actvn(self.weight_up4(w))
            w = self.weight_last(w)
            w = self.weight_final(w)
        if self.opt.use_restnet:
            for i in range(self.opt.n_blocks):
                resblock = getattr(self,'ResnetBlock'+str(i))
                x = resblock(x)
                # print(x.shape)
        # if self.opt.input_size>=256:
        #     x = self.actvn(self.up0(x))
        x = self.actvn(self.up1(x))
        x = self.actvn(self.up2(x))
        x = self.actvn(self.up3(x))
        if self.opt.input_size>=256:
            x = self.actvn(self.up4(x))
        x = self.last(x)
        return x,w
    def make_random_degree(self):
        # TODO remeber to add a to(device)
        pass

# **************************************************************************** #
class Encoder3(nn.Module):
    def __init__(self,opt):
        super(Encoder2, self).__init__()
        self.opt = opt
        inputCH = opt.input_chan*3 # current + last + next is fixed
        CH_list = 'current,last,next,'
        if opt.use_difference:
            inputCH += opt.input_chan # 12
            CH_list+='difference,'
        if opt.use_label:
            inputCH+=1 # 13
            CH_list+='segmap,'
        if opt.use_wireframe:
            inputCH += 1
            CH_list+='wireframe,'
        if opt.use_instance:
            inputCH += 1 # 14
            CH_list+='instance,'
        if opt.use_degree == 'wrt_position':
            inputCH += (opt.granularity+1)
            CH_list+='wrt_position,'
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ng = opt.firstK
        norm_layer = get_nonspade_norm_layer(opt,opt.norm_type)
        self.layer_0 = norm_layer(nn.Conv2d(input_CH, ndf, kw, stride=2, padding=pw))
        self.layer_1 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer_2 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer_3 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer_4 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.actvn = nn.LeakyReLU(0.2, False)
        coefficient = 128 if self.opt.input_size>=256 else 512
        self.fc_mu = nn.Sequential(*[nn.Linear(ng * coefficient, opt.z_dim)])
        self.fc_var = nn.Sequential(*[nn.Linear(ng * coefficient, opt.z_dim)])
        print('Encoder is using:' + CH_list)
    def forward(self,input):
        x = input
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.actvn(self.layer_0(x))
        x = self.actvn(self.layer_1(x))
        x = self.actvn(self.layer_2(x))
        x = self.actvn(self.layer_3(x))
        x = self.actvn(self.layer_4(x))
        if self.opt.input_size >= 256:
            x = self.actvn(self.layer_5(x))
        # x = self.actvn(x) # don't know why need to pass to 2 actvn...
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar
# **************************************************************************** #
class Deocder3(nn.Module):
    def __init__(self,opt,input_CH):
        super(Deocder, self).__init__()
        self.opt = opt
        self.sw = opt.input_size // (2**opt.n_downsample_global)
        self.sh = self.sw
        self.fc = nn.Sequential(*[nn.Linear(opt.z_dim, 8 * opt.firstK * self.sw * self.sh),nn.LeakyReLU(True)])
        # self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        # norm_layer = get_nonspade_norm_layer(opt, opt.norm_type)

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
        self.actvn = nn.LeakyReLU(0.2, False)
        # this is for extract the feature then cat with the above trans-conv
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.firstK
        norm_layer = get_nonspade_norm_layer(opt,opt.norm_type)

        cat_CH = input_CH
        # last current +  one-hot + (edge + segmap)
        self.layer1 = norm_layer(nn.Conv2d(cat_CH, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        if not self.opt.use_raw_only:
          self.w_head = copy.deepcopy(self.head)
          self.w_convtan1 = copy.deepcopy(self.convtan1)
          self.w_convtan2 = copy.deepcopy(self.convtan2)
          self.w_convtan3 = copy.deepcopy(self.convtan3)
          # last is only 1CH
          self.w_last = nn.Sequential(*[nn.ConvTranspose2d(in_channels=opt.firstK,
                                              out_channels=1,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),nn.BatchNorm2d(opt.output_channel)])
    # **********************************WEIGHT CONV LAYER **********************************
    def forward(self, cat_frame,input=None):
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
        img_feature = skip
        skips.append(skip)
        skips.reverse()
        weight_skips = skips
        x = self.fc(input)
        x = x.view(-1, 8 * self.opt.firstK, self.sw , self.sh)
        weight_x = x
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
        if not self.opt.use_raw_only:
          weight_x = self.actvn(weight_x) 
          weight_x = torch.cat((weight_x, weight_skips.pop(0)), dim=1)
          weight_x = self.head(weight_x)
          weight_x = self.actvn(weight_x)
          weight_x = torch.cat((weight_x, weight_skips.pop(0)), dim=1)
          weight_x = self.convtan1(weight_x)
          weight_x = self.actvn(weight_x)
          weight_x = torch.cat((weight_x,weight_skips.pop(0)),dim=1)
          weight_x = self.convtan2(weight_x)
          weight_x = self.actvn(weight_x)
          weight_x = torch.cat((weight_x, weight_skips.pop(0)), dim=1)
          weight_x = self.convtan3(weight_x)
          weight_x = self.actvn(weight_x)
          weight_x = self.last(weight_x)
          w = self.actvn(weight_x)
        return x,w