import torch
import torch.nn as nn
import mydataprocess
from mydataprocess import dataset, mydataloader
import option
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import net.model

myoption = option.opt()
myoption.name = 'color'
myoption.which_epoch = '190'
myoption.mode = 'test'
myoption.load_from_drive = False
for name,value in vars(myoption).items():
    print('%s=%s'%(name,value))

mymodel = net.model.single_frame()
mymodel.initialize(myoption)

unloader = transforms.ToPILImage()  # reconvert into PIL image
def imshow(tensor,interval = 0.5):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save('./result/sketch_resultinference.png')

b = mymodel.inference('/home/waihinchan/Desktop/scar/dataset/step/_10/6.jpg')
imshow(b)




