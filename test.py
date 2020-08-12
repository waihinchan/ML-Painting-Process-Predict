import torch
import torch.nn as nn
import mydataprocess
from mydataprocess import dataset, mydataloader
import option
import model
from torchvision import transforms
from PIL import Image
import os




myoption = option.opt()
myoption.which_epoch = '40'
myoption.mode = 'test'
for name,value in vars(myoption).items():
    print('%s=%s'%(name,value))

mymodel = model.SCAR()
mymodel.initialize(myoption)


def grabdata(opt):
    path = os.path.join('./dataset',opt.name)
    inputname = path + '/a.png'
    print(inputname)
    input_image = Image.open(inputname)
    params = dataset.how_to_deal(opt,input_image.size)
    transforms_pipe = dataset.build_pipe(opt,params, method=Image.NEAREST, normalize=False)
    return transforms_pipe(input_image).unsqueeze(0)

theinput = grabdata(myoption)

imshow(mymodel.netG(theinput))