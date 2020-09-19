from mydataprocess import mydataloader
import option
import net.model
import os
import time
myoption = option.opt()
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

dataloader = mydataloader.Dataloader(myoption)
All_videos_frames = dataloader.load_data()

# ******************* fake data ************************* #
# fake_frames_num = 300
# all_frames = []
# for i in range(0,fake_frames_num):
#     all_frames+=[torch.rand(1,3,512,512)]
# target = all_frames[-1]
# input = {
#     'target':target,
#     'frames':all_frames
# }
# ******************* fake data ************************* #
mymodel = net.model.SCAR()
mymodel.initialize(opt = myoption)
print('start to train')
for i in range(1,mymodel.opt.epoch):
    epoch_start_time = time.time()
    for j, one_video_frames in enumerate(All_videos_frames,start=1):
        # the dataset will return a list of frame from only ONE video
        mymodel.set_requires_grad(mymodel.netD, True)
        mymodel.set_requires_grad(mymodel.netD_T, True)
        loss = mymodel(one_video_frames)
        # optmiz D nad D_T
        mymodel.optimizer_D.zero_grad()
        loss_D = loss['D_loss']
        loss_D.backward()
        mymodel.optimizer_D.step()
        mymodel.netD_T.zero_grad()
        loss_D = loss['D_T_loss']
        loss_D.backward()
        mymodel.optimizer_D_T.step()

        # before optmiz G, cool down the params of D and D_T

        mymodel.set_requires_grad(mymodel.netD, False)
        mymodel.set_requires_grad(mymodel.netD_T, False)

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
        print('epoch%s loss is %s' % (i,loss))

    if i % 10==0:
        print('%s_epoch_loss' % i)
        mymodel.save(i)


print('train finished')
