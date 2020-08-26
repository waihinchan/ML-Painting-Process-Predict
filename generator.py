import network
import Resnet
import torch.nn as nn
import torch
import torch.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
# ******************************* pix2pixHD global gen *************************************** #

"""
c7s1-64,d128,d256,d512,d1024,R1024,R1024,
R1024,R1024,R1024,R1024,R1024,R1024,R1024,
u512,u256,u128,u64,c7s1-3
"""

class pix2pixHD_Global_gen(nn.Module):
    def __init__(self,input_channel,K = 64 ,downsample_num = 4,padding_type = 'reflect',resnetblock_num = 9):
        super(pix2pixHD_Global_gen, self).__init__()
        global_network = []
        global_network+= [nn.ReflectionPad2d(3)]
        c7s164 = network.c7s1_k(input_channel,K)
        global_network += c7s164
        temp_K = K

        for i in range(1,downsample_num+1):
            ds = network.dk(temp_K,temp_K*2)
            temp_K*=2
            global_network+=ds


        for i in range(1,resnetblock_num+1):
            res = [Resnet.ResK(dim=temp_K,padding_type=padding_type,norm_layer=nn.InstanceNorm2d)]
            global_network += res

        for i in range(1,downsample_num+1):
            us = network.uk(input_channels=temp_K,stride=2,N=3,k=int(temp_K/2))
            temp_K = int(temp_K/2)
            global_network += us

        global_network += [nn.ReflectionPad2d(3)]
        c7s13 = network.c7s1_k(temp_K,input_channel)
        global_network += c7s13

        global_network = nn.Sequential(*global_network)
        self.model = global_network

    def forward(self,input):
        return self.model(input)


# ******************************* pix2pixHD global gen *************************************** #


# ******************************* pix2pix u-net ************************************ #

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

        last = [nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=3,stride=2,kernel_size=4,padding=1),nn.BatchNorm2d(3),nn.Tanh()]
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

# ******************************* pix2pix u-net ************************************ #

# ******************************* spade gen ************************************ #

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

class SpadeEncoder(nn.Module):
    """
    with N conv + 2 linear fc + actvn
    """
    def __init__(self,opt):
        super(SpadeEncoder, self).__init__()
        kernel_w = 3
        padding_w = int(np.ceil((kernel_w - 1.0) / 2))
        setattr(self,'layer_'+str(0),add_norm_layer(nn.Conv2d(opt.input_chan, opt.firstK, kernel_w, stride=2, padding=padding_w)))
        tempk = min(opt.firstK*2,256)
        # if first K = 256 or more, we restrict K*2 no more than 512
        for i in range(1,5):
            tempk = min(tempk*2,256)
            setattr(self,'layer_'+str(i),add_norm_layer(nn.Conv2d(tempk, tempk*2, kernel_w, stride=2, padding=padding_w)))
        if opt.inputsize > 256:
            self.layer5 = add_norm_layer(nn.Conv2d(tempk, tempk*2, kernel_w, stride=2, padding=padding_w))

        self.actvn = nn.LeakyReLU(0.2, False)
        self.so = s0 = 4
        # these are for return Distribution
        self.fc_mu = nn.Linear(opt.firstK * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(opt.firstK * 8 * s0 * s0, 256)
        # looks like a hard coding...
    def forward(self,input):
        x = input
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        # not understand why the cropsize much bigger than these one, but still need a interpolate?
        x = self.layer0(x)
        x = self.layer1(self.actvn(x))
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer5(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        # reshape
        # pass to 2 fc respectively
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        # one question, how to compute the loss?
        return mu, logvar

# https://blog.csdn.net/yangwangnndd/article/details/95490074
class SpadeGenerator(nn.Module):
    def __init__(self,opt,firstK = 64,input_noise_dim=256):
        # firstK = 64 can be instead in opt
        super(SpadeGenerator, self).__init__()
        nf = firstK
        self.input_noise_dim = input_noise_dim
        # z axis
        # for reshape
        self.up = nn.Upsample(scale_factor=2)
        # for upsample, link by each spaderesblock
        self.fc = nn.Linear(input_noise_dim, 16384)  # hardcoded
        # the h and w remain to count by the target size
        self.head_0 = Resnet.SpadeResBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = Resnet.SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = Resnet.SPADEResnetBlock(16 * nf, 16 * nf, opt)
    # up+spaderesblock
        tempk = 16*nf
        for i in range(opt.upsample_num):
            # 1024 512 256 128 64 64 64
            # need a test
            setattr(self,'up_'+str(i),Resnet.SpadeResBlock(ni = tempk, nf =int(tempk/2) , opt = opt))
            tempk = max(int(tempk/2),nf)
            print(getattr(self,'up_'+str(i)))
        # up+spaderesblock

        final_nc = int(tempk/2)
        # as tempk no less than nf, the last would no less than nf//2 (last output is tempk / 2 )
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        # output the last image(CH*W*H)

    def forward(self,input,z=None):
        seg = input
        # they use z as the code vector, the input is for the spade block y and b
        # if use encoder, we will use a image for encoder the style image then generate a Z, instead a random noise Z
        # this Z would match the style of the style image but match the seg map lay out
        if z is None:
            # create a batchsize num * Zdim  gaussian random tensor
            z = torch.randn(input.size(0), self.opt.z_dim,
                            dtype=torch.float32, device=input.get_device())
        x = self.fc(z)
        # bs*(sh*sw*CH*16)(16*CH = 16*nf nf = first K)
        x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        # Bs*sh*sw*CH*16
        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        if self.opt.upsample_num>=5:
            x = self.up(x)
        x = self.G_middle_1(x, seg)
        x = self.up(x)
        for i in range(self.opt.upsample_num):
            i_up = getattr(self,'up_'+str(i))
            x = i_up(x)
            x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x
        # make a randomZ, gothouh from a fc, then pass into the generator to generate a image, different would generate
        # different style, so Z is the style, the seg is out the layout would be?

# ******************************* spade gen ************************************ #


# some test #

# testforward = pix2pixHD_gen(padding_type='reflect')
# print(testforward(torch.rand(1,3,1024,1024)).shape)

# unet = pix2pix_generator(3)
# print('upstack')
# print(unet.upstack)
# print('downstack')
# print(unet.downstack)
# print(unet.last)
# inputx = torch.rand(1,3,1024,1024)
# print(unet(inputx).shape)
# a = torch.rand(1,3,1024,1024)
# b = torch.rand(1,3,1024,1024)
# c = torch.cat((a,b),1)
# print(c.shape)

# some test #