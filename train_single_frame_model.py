# this is for train the color image to sketch
# next we need to

import option
import net.model
import os
import time
from mydataprocess import mydataloader
myoption = option.opt()
# myoption.save_result=True
myoption.name = 'color'
myoption.mode = 'train'
# myoption.which_epoch = '190'
myoption.batchSize = 5
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

dataloader = mydataloader.Dataloader(myoption)
All_data = dataloader.load_data()

mymodel = net.model.single_frame()
# is single_frame but we take data from
mymodel.initialize(opt = myoption)
print('start to train')
for i in range(1,mymodel.opt.epoch):
    epoch_start_time = time.time()
    for j, one_video_frames in enumerate(All_data,start=1):
        mymodel.set_requires_grad(mymodel.netD, True)
        loss = mymodel(one_video_frames)
        mymodel.optimizer_D.zero_grad()
        loss_D = loss['D_loss']
        loss_D.backward()
        mymodel.optimizer_D.step()
        mymodel.set_requires_grad(mymodel.netD, False)
        G_loss = loss['G_loss']
        mymodel.optimizer_G.zero_grad()
        G_loss.backward()
        mymodel.optimizer_G.step()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (i, myoption.epoch, time.time() - epoch_start_time))
    if i >= mymodel.opt.niter_decay:
        # if epoch less than the opt.niter_decay, this won't happen
        # 0.1**(current_epoch - decay_epoch // 30)
        # (i)160 - 100 = 60. 60 // 30 = 2 0.1 **2 = 0.01 , decay 10% every 30 epoch
        updateepoch = i - mymodel.opt.niter_decay
        mymodel.update_learning_rate(updateepoch)
    if i % 5 == 0:
        print('epoch%s last loss is %s' % (i, loss))
    if i % 10==0:
        mymodel.save(i)
        print('save %s_epoch' % i)
        mymodel.opt.save_result = True
    if i % 11 == 1:
        mymodel.opt.save_result = False
print('train finished')
