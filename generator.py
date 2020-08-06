import network
import Resnet
import torch.nn as nn
import torch

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
        # print('input_size')
        # print(x.shape)
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