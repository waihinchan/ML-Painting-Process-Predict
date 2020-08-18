from mydataprocess import mydataloader
import option
import model
from utils import visual
import os
import time
myoption = option.opt()
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

dataloader = mydataloader.Dataloader(myoption)

thedataset = dataloader.load_data()


print('create a dataset with %s group' % dataloader.__len__())

# data = {'image':torch.rand(1,3,512,512),'label':torch.rand(1,3,512,512)}

mymodel = model.SCAR()
mymodel.initialize(myoption)


tflogpath = './run/' + myoption.name
if os.path.isdir(tflogpath):
    for i in os.listdir(tflogpath):
        os.remove(os.path.join(tflogpath,i))

    print('tflog been clean!')

myvisualer = visual.Visualizer(tflogpath)
testpath = os.path.join(os.path.join(os.getcwd(), 'dataset'), myoption.name) + '/test'

loss_item = {'G_loss':0,'D_loss':0}

for i in range(1,mymodel.opt.epoch):

    epoch_start_time = time.time()

    for j, data in enumerate(thedataset,start=j):
        theinputdata = data
        # theinputdata = {'label':data['step_1'],'image':data['target']}
        mymodel.set_requires_grad(mymodel.netD, True)
        loss = mymodel(theinputdata)

        # optmiz D
        mymodel.optimizer_D.zero_grad()
        loss_D = loss['dis_loss']
        loss_item['D_loss'] += loss_D
        loss_D.backward()
        mymodel.optimizer_D.step()

        # before optmiz G, cool down the params of D
        # this part reamain to further update to
        # separate the learning rate and params of each part of the models
        mymodel.set_requires_grad(mymodel.netD, False)

        G_loss = loss['G_loss']
        loss_item['G_loss'] += G_loss
        mymodel.optimizer_G.zero_grad()
        # G_loss.backward(retain_graph=True)
        G_loss.backward()
        mymodel.optimizer_G.step()

        if j % 100 == 0:
            print('the last average 50 iter Dloss is %s'%(loss_item['D_loss']))
            print('the last average 50 iter Gloss is %s'%(loss_item['G_loss']))
            loss_item['D_loss'] = loss_item['D_loss'] / 100
            loss_item['G_loss'] = loss_item['G_loss'] / 100
            loss_item['G_loss'] = 0
            loss_item['D_loss'] = 0


    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (i, myoption.epoch, time.time() - epoch_start_time))

    if i >= mymodel.opt.niter_decay:
        # if epoch less than the opt.niter_decay, this won't happen
        # 0.1**(current_epoch - decay_epoch // 30)
        # (i)160 - 100 = 60. 60 // 30 = 2 0.1 **2 = 0.01 , decay 10% every 30 epoch
        updateepoch = i - mymodel.opt.niter_decay
        mymodel.update_learning_rate(updateepoch)
        # this can be update for more elegant

    myvisualer.visulize_loss(epoch=i ,loss_dict=loss)
    # not sure why some loss not show
    if i % 5 == 0:
        print('epoch%s loss is %s' % (i,loss))

    if i % 10==0:
        print('%s_epoch_loss' % i)
        mymodel.save(i)
        # not sure this working , remain to test
        myvisualer.get_result(path=testpath,epoch=i,netG = mymodel.netG)
        # not sure this working , remain to test




