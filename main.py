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
# testpath = visual.get_test_data('scar/dataset/test')
epoch = 100


for i in range(epoch):
    for data in thedataset:
        theinputdata = {'label':data['step_1'],'image':data['target']}
        loss = mymodel(theinputdata)

        # optmiz D
        mymodel.set_requires_grad(mymodel.netD, True)
        mymodel.optimizer_D.zero_grad()
        loss_D = loss['dis_loss']
        loss_D.backward()
        mymodel.optimizer_D.step()
        print('optmiz D')
        # optmiz G
        mymodel.set_requires_grad(mymodel.netD, False)
        # cooldown the params of netD
        G_loss = loss['G_loss']
        mymodel.optimizer_G.zero_grad()
        # G_loss.backward(retain_graph=True)
        G_loss.backward()
        mymodel.optimizer_G.step()
        print('optmiz G')


    if i%5==0:
        # myvisualer.visulize_loss(loss,i)
        print(loss)

    if i%10==0:
        mymodel.save(i)
        # print('%s_epoch_loss_%s' % (i,G_loss))
        # myvisualer.visulize_loss(loss,i)




