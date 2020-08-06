import torch
import torch.nn as nn

class ft(nn.Module):
    def __init__(self):
        super(ft, self).__init__()
        self.model = []
        self.model+= [nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3),nn.ReLU,nn.Conv2d(in_channels=6,out_channels=12,kernel_size=3)]
        self.net = nn.Sequential(*self.model)

    def forward(self,input):

        return self.net(nn.LeakyReLU(input))

me = ft()
print(me.parameters())

# 笔记
# functional 是没有学习参数的东西用的
# 而 nnmodule可以自动追踪参数
# 有些时候可以直接写吧。。。分得太散还更麻烦