import torch
import torch.nn as nn
import mydataprocess
from mydataprocess import dataset, mydataloader
import option
import model
from torchvision import transforms
import matplotlib.pyplot as plt
myoption = option.opt()
for name,value in vars(myoption).items():
    print('%s=%s'%(name,value))

dataloader = mydataloader.Dataloader(myoption)

thedataset = dataloader.load_data()

print('create a dataset with %s group' % dataloader.__len__())

data = {'image':torch.rand(1,3,512,512),'label':torch.rand(1,3,512,512)}
# fake data for test

mymodel = model.SCAR()
mymodel.initialize(myoption)

epoch = 100

for i in range(epoch):
    for data in thedataset:
        theinputdata = {'label':data['step_2'],'image':data['target']}
        loss = mymodel(theinputdata)
        # optmiz G
        G_loss = loss['G_loss']
        mymodel.optimizer_G.zero_grad()
        G_loss.backward(retain_graph=True)
        mymodel.optimizer_G.step()
        # optmiz D
        mymodel.optimizer_D.zero_grad()
        loss_D = loss['dis_loss']
        loss_D.backward()
        mymodel.optimizer_D.step()

    if i%10==0:
        mymodel.save(i)
        print('%s_epoch_loss_%s' % (i,G_loss))
#
