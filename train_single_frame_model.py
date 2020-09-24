import option
import net.model
import os
import time
from mydataprocess import mydataloader
myoption = option.opt()
# myoption.save_result=True
myoption.name = 'color'
myoption.mode = 'continue_train'
myoption.which_epoch = '190'
myoption.batchSize = 5
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

dataloader = mydataloader.Dataloader(myoption)
All_data = dataloader.load_data()

mymodel = net.model.single_frame()
mymodel.initialize(opt = myoption)


print_loss = {
    "G_loss": 0,
    "D_loss": 0,
    "vgg_loss": 0,
    "l1_loss": 0,
    "gan_loss": 0,
    "TV_loss": 0
}

print('start to train')
for i in range(1,mymodel.opt.epoch):
    epoch_start_time = time.time()
    for j, one_video_frames in enumerate(All_data,start=1):
        # the dataset will return a list of frame from only ONE video
        mymodel.set_requires_grad(mymodel.netD, True)
        loss = mymodel(one_video_frames)
        # optmiz D
        mymodel.optimizer_D.zero_grad()
        loss_D = loss['D_loss']
        loss_D.backward()
        mymodel.optimizer_D.step()
        mymodel.set_requires_grad(mymodel.netD, False)
        G_loss = loss['G_loss']
        mymodel.optimizer_G.zero_grad()
        G_loss.backward()
        mymodel.optimizer_G.step()
        print_loss['G_loss'] += G_loss
        print_loss['D_loss'] += loss_D
        print_loss['vgg_loss'] += loss['vgg_loss']
        print_loss['l1_loss'] += loss['l1_loss']
        print_loss['gan_loss'] += loss['gan_loss']
        print_loss['TV_loss'] += loss['TV_loss']

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (i, myoption.epoch, time.time() - epoch_start_time))
    if i >= mymodel.opt.niter_decay:
        # if epoch less than the opt.niter_decay, this won't happen
        # 0.1**(current_epoch - decay_epoch // 30)
        # (i)160 - 100 = 60. 60 // 30 = 2 0.1 **2 = 0.01 , decay 10% every 30 epoch
        updateepoch = i - mymodel.opt.niter_decay
        mymodel.update_learning_rate(updateepoch)
    if i % 5 == 0:
        for each_loss in print_loss.keys():
            print_loss[each_loss] = print_loss[each_loss] / 5 / j
        print('epoch%s average loss is %s' % (i,print_loss))
        print('epoch%s last loss is %s' % (i, loss))
        mymodel.opt.save_result = True
        # reset the print loss
        print_loss = {
            "G_loss": 0,
            "D_loss": 0,
            "vgg_loss": 0,
            "l1_loss": 0,
            "gan_loss": 0,
            "TV_loss": 0
        }
        # reset the print loss
    if i % 5 ==1:
        mymodel.opt.save_result = False
    if i % 10==0:
        mymodel.save(i)
        print('save %s_epoch' % i)
print('train finished')
