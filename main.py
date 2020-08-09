import torch
import torch.nn as nn
from mydataprocess import dataset, mydataloader
import option
import model


myoption = option.opt()
for name,value in vars(myoption).items():
    print('%s=%s'%(name,value))
data = {'image':torch.rand(1,3,512,512),'label':torch.rand(1,3,512,512)}
mymodel = model.SCAR()
mymodel.initialize(myoption)


for i in range(3):
    loss = mymodel(data)
    print(loss)
    # optmiz G
    total_loss = loss['total_loss']
    mymodel.optimizer_G.zero_grad()
    total_loss.backward(retain_graph=True)
    mymodel.optimizer_G.step()
    # optmiz D
    mymodel.optimizer_D.zero_grad()
    loss_D = loss['dis_loss']
    loss_D.backward()
    mymodel.optimizer_D.step()
    mymodel.save(i)

