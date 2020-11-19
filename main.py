#gittest
from mydataprocess import mydataloader
import option
import net.model
import time
import os
if not os.path.isdir('./result/result_preview'):
  os.mkdir('./result/result_preview')
myoption = option.opt()
myoption.batchSize = 1 # if we have a batch norm maybe this would still working?
myoption.name = 'pair'
myoption.use_degree = 'wrt_position'
myoption.use_label= True
myoption.mode = 'continue train'
myoption.which_epoch = 400
myoption.forward = 'seq'
myoption.Kld_lambda = myoption.Kld_lambda if myoption.mode == 'train' else (1.1**(myoption.which_epoch//20))*myoption.Kld_lambda
myoption.learningrate = myoption.learningrate if myoption.mode == 'train' and myoption.which_epoch<myoption.niter_decay else myoption.learningrate * (0.9 ** ( (myoption.which_epoch-myoption.niter_decay) // 10))

for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))
dataloader = mydataloader.Dataloader(myoption)
pair_data = dataloader.load_data()
mymodel = net.model.SCAR()
mymodel.initialize(opt = myoption)
start_epoch = 1 if myoption.mode == 'train' else myoption.which_epoch
# start_epoch = 1 
print('start to train')
for i in range(start_epoch,mymodel.opt.epoch):
    epoch_start_time = time.time()
    # print_loss = {'G_loss':0,'D_loss':0}
    for j, pair in enumerate(pair_data,start=1):
        loss,fake = mymodel(pair,myoption.forward)
        mymodel.set_requires_grad(mymodel.netD, False)
        G_loss = loss['G_loss']
        # print_loss[G_loss]+=G_loss
        mymodel.optimizer_G.zero_grad()
        G_loss.backward()
        mymodel.optimizer_G.step()
        mymodel.set_requires_grad(mymodel.netD, True)
        mymodel.optimizer_D.zero_grad()
        loss_D = loss['D_loss']
        # print_loss[D_loss]+=loss_D
        loss_D.backward()
        mymodel.optimizer_D.step()
    # print_loss['G_loss']=print_loss['G_loss']/j
    # print_loss['D_loss']=print_loss['D_loss']/j
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (i, myoption.epoch, time.time() - epoch_start_time))
    if i >= mymodel.opt.niter_decay:
        updateepoch = i - mymodel.opt.niter_decay
        mymodel.update_learning_rate(updateepoch)
    if i % 5 == 0:
        print('epoch%s last loss is %s' % (i, loss))
        # print('epoch%s average loss is %s' % (i, print_loss))
    if i % 5 == 1:
        print('epoch%s last loss is %s' % (i, loss))
        mymodel.opt.save_result = False
        print('current kld lambda is %s' %  mymodel.opt.Kld_lambda)
    if i%20 == 0:
        mymodel.opt.Kld_lambda =  mymodel.opt.Kld_lambda + mymodel.opt.Kld_lambda * 0.1

    if i % 50 == 0:
        mymodel.save(i)
        print('save %s_epoch' % i)
    # if i % 50 == 0:
    #     mymodel.opt.save_result = True
    if i % 100 == 0:
        mymodel.opt.save_result = True
    if i % 100 == 1:
        mymodel.opt.save_result = False
    # mymodel.opt.save_result = True
print('train finished')

