from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torch
from mydataprocess.dataset import is_image_file
import time


unloader = transforms.ToPILImage()  # reconvert into PIL image
def imshow(tensor,interval = 0.5):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.figure()
    plt.imshow(image)
    plt.show()
    plt.pause(interval)
    plt.close('all')


# while runnning get the result
def get_test_data(path):
    assert os.path.isdir(path),print('test folder not exist!')
    dir = sorted(os.path.join(path , i) for i in os.listdir(path) if is_image_file(i) )
    return dir

def save_result(epoch,tensor,name,path = './result'):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    name = '%sepoch_%s' % (epoch,name)
    image.save(os.path.join(path, name))

class Visualizer():
    def __init__(self,path):
        self.writer = SummaryWriter(path)
        """
        loss_dict = {'name1':loss1,'name2':loss2...}
        """
    def visulize_loss(self,loss_dict,epoch):
        for name in loss_dict.keys():
            self.writer.add_scalar(name,loss_dict[name],epoch)
            print(name,str(loss_dict[name]))

    def get_result(self,path,netG,epoch):
        testimages = get_test_data(path)
        pipe = []
        pipe.append(transforms.ToTensor())
        transform_pipe = transforms.Compose(pipe)
        resultfolder = path + '/epoch' + str(epoch)
        print(resultfolder)
        os.makedirs(resultfolder)
        print('saving result of %s_epoch, total %s image' % (epoch,len(testimages)))
        for testimage in testimages:
            rawimage = Image.open(testimage)
            w,h = rawimage.size
            rawtensor = transform_pipe(rawimage)[:,:,:int(w/2)].unsqueeze(0)
            result = netG(rawtensor.to("cuda" if torch.cuda.is_available() else "cpu"))
            pathlist = testimage.split('/',-1)
            save_result(epoch=epoch,tensor=result,name=pathlist[-1],path=resultfolder)
        print('Done')



