import torch
import torch.nn as nn
from mydataprocess import dataset, mydataloader
import generator
import Loss
import discriminator
import option
epochs = 10
input_chan = 1
myG = generator.pix2pix_generator(input_channel=input_chan)
myD = discriminator.patchGAN(input_channel=4)
myopt = option.opt()

for name,value in vars(myopt).items():
    print('%s=%s'%(name,value))
myloss = Loss.GANLoss()
dataloader = mydataloader.Dataloader(myopt)
dataset = dataloader.load_data()
optimizer_G = torch.optim.Adam(myG.parameters(),lr=1e-4)
optimizer_D = torch.optim.Adam(myD.parameters(),lr=1e-4)

for epoch in range(epochs):
    for data in dataset:
        generated = myG(data['label'])
        cat_fake = torch.cat((generated,data['label']),1)
        cat_real = torch.cat((data['image'],data['label']),1)

        dis_real = myD(cat_real)
        dis_fake = myD(cat_fake)
        loss_real = myloss(dis_real,True)
        loss_fake = myloss(dis_fake,True)
        dis_loss = loss_real + loss_fake

        gan_loss = myloss(generated,True)

        l1_loss_object = nn.L1Loss()
        l1_loss = l1_loss_object(generated,data['image'])
        gen_total_loss = gan_loss + 100 * l1_loss

        print('dis_loss'+str(dis_loss))
        print('gen_total_loss'+str(gen_total_loss))

        optimizer_G.zero_grad()
        gen_total_loss.backward(retain_graph=True)
        optimizer_G.step()

        optimizer_D.zero_grad()
        dis_loss.backward()
        optimizer_D.step()





        # print(loss_real)
        # print(loss_fake)
