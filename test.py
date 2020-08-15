import torch
import torch.nn as nn
import mydataprocess
from mydataprocess import dataset, mydataloader
import option
import model
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt



myoption = option.opt()
myoption.which_epoch = '60'
myoption.mode = 'test'
for name,value in vars(myoption).items():
    print('%s=%s'%(name,value))

mymodel = model.SCAR()
mymodel.initialize(myoption)


def grabdata(opt,path):
    # path = os.path.join('./dataset',opt.name)
    # inputname = path + '/a.png'
    # print(inputname)
    input_image = Image.open(path)
    # transforms_pipe = dataset.build_pipe(opt)
    pipe = []
    pipe.append(transforms.ToTensor())
    pipe = transforms.Compose(pipe)
    image = pipe(input_image)[:,:,512:]
    # image = pipe(input_image)

    return image.unsqueeze(0)



unloader = transforms.ToPILImage()  # reconvert into PIL image
def imshow(tensor,interval = 0.5):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save('./result/result.png')


a = grabdata(myoption,'./dataset/test/haha.jpg').to(mymodel.device)
# print(a.shape)
# print(torch.cat((a,a),1).shape)
b = mymodel.netG(a)
imshow(b)




