import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Loss import gram
class ft(nn.Module):
    def __init__(self):
        super(ft, self).__init__()
        self.model = []
        model+=[]
    def forward(self,input):
        return self.net(input)


# me = ft()
# for i in me.parameters():
#     print(i.shape)

# 笔记
# functional 是没有学习参数的东西用的
# 而 nnmodule可以自动追踪参数
# 有些时候可以直接写吧。。。分得太散还更麻烦

# x = torch.tensor(1.0, requires_grad=True)
# y = torch.tensor(2.0, requires_grad=True)
x = torch.rand(1,2,3,4)
y = torch.rand(1,2,3,4).requires_grad_()
optimizer = optim.LBFGS([y])

now = [0]
total = 200
while now[0] < total:
    def closure():
        optimizer.zero_grad()
        z = F.mse_loss(gram(x),gram(y))*100
        z.backward()
        now[0] += 1
        print(y)
        return z
    optimizer.step(closure)
