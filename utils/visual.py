from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torch

import time
# this is for visualize the result
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
    # here remain to update
    # like how to inline the result

# while runnning get the result
def get_test_data(path):
    assert os.path.isdir(path),print('test folder not exist!')
    dir = sorted(os.path.join(path,i) for i in os.listdir(path))
    return dir

def save_result(epoch,tensor,name,path = './result'):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    name = '%s_%s_%s.png' % (name,epoch,time.time())
    image.save(os.path.join(path, name))

class Visualizer():
    def __init__(self,path):
        self.writer = SummaryWriter(path)
        """
        loss_dict = {'name1':loss1,'name2':loss2...}
        """
    def visulize_loss(self,loss_dict,wrt,epoch):
        for name in loss_dict.keys():
            self.writer.add_scalar(name+wrt,loss_dict[name],epoch)










