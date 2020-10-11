from mydataprocess import mydataloader
import option
import net.model
import os
import time
myoption = option.opt()
myoption.batchSize = 1
myoption.input_size = 256
myoption.use_difference = False
# myoption.mode = 'continue train'
# myoption.which_epoch = 190
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

dataloader = mydataloader.Dataloader(myoption)
pair_data = dataloader.load_data()

mymodel = net.model.pair_frame_generator()
mymodel.initialize(opt = myoption)

import torch
print('start to train')
for i in range(1,mymodel.opt.epoch):
    epoch_start_time = time.time()
    for j, pair in enumerate(pair_data,start=1):
        loss = mymodel(pair)
        mymodel.set_requires_grad(mymodel.netD, False)
        G_loss = loss['G_loss']
        mymodel.optimizer_G.zero_grad()
        G_loss.backward()
        mymodel.optimizer_G.step()
        mymodel.set_requires_grad(mymodel.netD, True)
        mymodel.optimizer_D.zero_grad()
        loss_D = loss['D_loss']
        loss_D.backward()
        mymodel.optimizer_D.step()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (i, myoption.epoch, time.time() - epoch_start_time))
    if i >= mymodel.opt.niter_decay:
        updateepoch = i - mymodel.opt.niter_decay
        mymodel.update_learning_rate(updateepoch)
    if i % 5 == 0:
        print('epoch%s last loss is %s' % (i, loss))
        mymodel.opt.save_result = True
    if i % 10 == 0:
        mymodel.save(i)
        print('save %s_epoch' % i)
print('train finished')


# from mydataprocess import mydataloader
# import option
# import net.model
# import os
# import time
# myoption = option.opt()
# for name,value in vars(myoption).items():
#     print('%s=%s' % (name,value))
#
# dataloader = mydataloader.Dataloader(myoption)
# pair_data = dataloader.load_data()
#
# mymodel = net.model.time_scar()
# mymodel.initialize(opt = myoption)
# print('model structure')
# print(mymodel)
# print('start to train')
# for i in range(1,mymodel.opt.epoch):
#     epoch_start_time = time.time()
#     for j, pair in enumerate(pair_data,start=1):
#         loss = mymodel(pair)
#         G_loss = loss['G_loss']
#         mymodel.optimizer_G.zero_grad()
#         G_loss.backward()
#         mymodel.optimizer_G.step()
#     print('End of epoch %d / %d \t Time Taken: %d sec' %
#           (i, myoption.epoch, time.time() - epoch_start_time))
#     if i >= mymodel.opt.niter_decay:
#         updateepoch = i - mymodel.opt.niter_decay
#         mymodel.update_learning_rate(updateepoch)
#     if i % 5 == 0:
#         print('epoch%s last loss is %s' % (i, loss))
#         mymodel.opt.save_result = True
#     if i % 5 == 1:
#         mymodel.opt.save_result = False
#     if i % 10 == 0:
#         mymodel.save(i)
#         print('save %s_epoch' % i)
# print('train finished')