import network
import torch
import torch.nn as nn

class patchGAN(nn.Module):
    def __init__(self,input_channel,K = 64,n_layers = 4):
        super(patchGAN, self).__init__()
        model = []
        model += [nn.Conv2d(input_channel, K, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        temp_k = K

        for i in range(1, n_layers):
            # the first one is at the top
            # not over than 512 filters
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


# some test

# input = torch.rand(1,3,1024,1024)
# ma = patchGAN()
# print(ma(input).shape)
# print(ma)

# some test
