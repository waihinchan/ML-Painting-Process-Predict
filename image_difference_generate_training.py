from mydataprocess import mydataloader
import option
import net.model
import os
import time
myoption = option.opt()
myoption.batchSize = 1
# if we have a batch norm maybe this would still working?
myoption.input_size = 256
myoption.name = 'pair'
myoption.use_label= True
myoption.mode = 'continue train'
myoption.which_epoch = 400
myoption.forward = 'seq'
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

dataloader = mydataloader.Dataloader(myoption)
pair_data = dataloader.load_data()

mymodel = net.model.label_VAE()
mymodel.initialize(opt = myoption)

print('start to train')
for i in range(1,mymodel.opt.epoch):
    epoch_start_time = time.time()
    for j, pair in enumerate(pair_data,start=1):
        loss,fake = mymodel(pair,myoption.forward)
        mymodel.set_requires_grad(mymodel.netD, False)
        G_loss = loss['G_Loss']
        mymodel.optimizer_G.zero_grad()
        G_loss.backward()
        mymodel.optimizer_G.step()
        mymodel.set_requires_grad(mymodel.netD, True)
        mymodel.optimizer_D.zero_grad()
        loss_D = loss['D_Loss']
        loss_D.backward()
        mymodel.optimizer_D.step()

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (i, myoption.epoch, time.time() - epoch_start_time))
    if i >= mymodel.opt.niter_decay:
        updateepoch = i - mymodel.opt.niter_decay
        mymodel.update_learning_rate(updateepoch)
    if i % 5 == 0:
        print('epoch%s last loss is %s' % (i, loss))
    if i % 5 == 1:
        print('epoch%s last loss is %s' % (i, loss))
        mymodel.opt.save_result = False
    if i % 50 == 0:
        mymodel.save(i)
        print('save %s_epoch' % i)
    if i % 50 == 0:
        if i >= 50:
            mymodel.opt.save_result = True
    if i % 10 == 1:
        mymodel.opt.save_result = False
print('train finished')

# TODO 写一个新的make pair，把所有粒度都涵盖进去
#  粒度的思路在于 我们现在有一个segmap 和 一个 one-hot，
#  如果说one-hot不是固定，那结构就是CVAE，如果one-hot是固定就不存在CVAE了，
#  所以在CVAE的情况下，我们的one-hot是随机的/或者是用另外一个randomZ去做一个分类，这意味着在inference的时候这个C是随机的
#  同时在训练初期我们有一个segmap 是不完整的 ，但是同样在inference的时候segmap是随机的，虽然可以用硬编码的方式来做随机组合的segmap
#  但是即使在这种情况下也是不完整的，尤其是初期模型没有学习过某些label缺失的情况下如何生成 所以粒度应该是
#  从小粒度到大粒度都涵盖到，比如说每帧之间成对，然后扩大帧与帧之间的步长再成对
#  但是前期还是有缺失的 两个方案 一个是做一个mask，
#  另外一个就是强行找到步长能够覆盖所有的segmap的时候再做分组
#  （这个方法可以单独对某个数据做，但是对于粒度不同的数据 处理是很麻烦的，要手动去做一个检查点来划分确认哪一个是有完整segmap的）
#  对于第一个mask的方案，如果用segmap分割区域然后单独训练是不是也可以 其实现在做的也是类似这样的但是没有明确分割开
