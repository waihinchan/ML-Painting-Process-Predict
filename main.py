from mydataprocess import mydataloader
import option
import model
from utils import visual


myoption = option.opt()
for name,value in vars(myoption).items():
    print('%s=%s'%(name,value))

dataloader = mydataloader.Dataloader(myoption)

thedataset = dataloader.load_data()


print('create a dataset with %s group' % dataloader.__len__())

# data = {'image':torch.rand(1,3,512,512),'label':torch.rand(1,3,512,512)}


mymodel = model.SCAR()
mymodel.initialize(myoption)
tflogpath = './run/' + myoption.name
myvisualer = visual.Visualizer(tflogpath)
testpath = visual.get_test_data('scar/dataset/test')



for i in range(mymodel.opt.epoch):
    loss_item = {'G_loss':0,'D_loss':0}
    for iter, data in enumerate(thedataset):
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

        # optmiz G
        mymodel.set_requires_grad(mymodel.netD, False)

        G_loss = loss['G_loss']
        loss_item['G_loss'] += G_loss
        mymodel.optimizer_G.zero_grad()
        # G_loss.backward(retain_graph=True)
        G_loss.backward()
        mymodel.optimizer_G.step()

        # need test
        if iter % 50 == 0:
            loss_item['D_loss'] = loss_item['D_loss'] / 50
            loss_item['G_loss'] = loss_item['G_loss'] / 50
            myvisualer.writer.add_scalar('G_loss',loss_item['G_loss'],iter)
            myvisualer.writer.add_scalar('D_loss',loss_item['D_loss'],iter)
            loss_item['G_loss'] = 0
            loss_item['D_loss'] = 0
        # need test
    if i >= mymodel.opt.niter_decay:
        # if epoch less than the opt.niter_decay, this won't happen
        # 0.1**(current_epoch - decay_epoch // 30)
        # (i)160 - 100 = 60. 60 // 30 = 2 0.1 **2 = 0.01 , decay 10% every 30 epoch
        updateepoch = i - mymodel.opt.niter_decay
        mymodel.update_learning_rate(updateepoch)
        # this can be update for more elegant


    if i%5==0:
        print('%s loss is %s' % (i,loss))

    if i%9==0:
        mymodel.save(i)
        print('%s_epoch_loss_%s' % (i,loss))
        myvisualer.visulize_loss(loss=loss,wrt='epoch',epoch=i)




