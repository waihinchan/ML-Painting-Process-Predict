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
print_loss = {
    "G_loss": 0,
    "D_loss": 0,
    "vgg_loss": 0,
    "l1_loss": 0,
    "gan_loss": 0,
    "firstG_loss": 0 if mymodel.opt.generate_first_frame else None
}
print('start to train')
for i in range(1,mymodel.opt.epoch):
    epoch_start_time = time.time()
    for j, one_video_frames in enumerate(All_videos_frames,start=1):
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
        if mymodel.opt.generate_first_frame:
            firstG_loss = loss['firstG_loss']
            mymodel.optimizer_G_first.zero_grad()
            firstG_loss.backward()
            mymodel.optimizer_G_first.step()

        print_loss['G_loss'] += G_loss
        print_loss['D_loss'] += loss_D
        print_loss['vgg_loss'] += loss['vgg_loss']
        print_loss['l1_loss'] += loss['l1_loss']
        print_loss['gan_loss'] += loss['gan_loss']
        print_loss['firstG_loss'] += loss['firstG_loss'] if mymodel.opt.generate_first_frame else None

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
        # reset the print loss
        for each_loss in print_loss.keys():
            print_loss[each_loss] = 0
        # reset the print loss
    if i % 5 ==1:
        mymodel.opt.save_result = False
    if i % 10==0:
        mymodel.save(i)
        print('save %s_epoch' % i)
print('train finished')
