import option
import net.model
import os
import time
from mydataprocess import mydataloader

# # this is for the wireframe generator
# wireframe_option = option.opt()
# wireframe_option.name = 'color'
# wireframe_option.which_epoch = '190'
# wireframe_option.mode = 'test'
# wireframe_model = net.model.single_frame()
# wireframe_model.initialize(wireframe_option)
# # this is for the wireframe generator


myoption = option.opt()
myoption.name = 'color'
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

dataloader = mydataloader.Dataloader(myoption)
All_data = dataloader.load_data()
#
mymodel = net.model.style_transfer_model()
mymodel.initialize(opt = myoption)
print(mymodel)
print('start to train')

for i in range(1,mymodel.opt.epoch):
    epoch_start_time = time.time()
    for j, pair_data in enumerate(All_data,start=1):
        # input_sketch = wireframe_model.inference(pair_data['input'])  # this is the "fake" input..
        # x = {
        #     'input':input_sketch,
        #     'target':pair_data['target']
        # }
        x = pair_data

        # mymodel.set_requires_grad(mymodel.netD, True)
        loss = mymodel(x) # not sure if need a detach
        # mymodel.optimizer_D.zero_grad()
        # loss_D = loss['D_loss']
        # loss_D.backward()
        # mymodel.optimizer_D.step()
        # mymodel.set_requires_grad(mymodel.netD, False)
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
    if i % 10 == 0:
        mymodel.save(i)
        print('save %s_epoch' % i)
        mymodel.opt.save_result = True
    if i % 11 == 0:
        mymodel.opt.save_result = False
print('train finished')
