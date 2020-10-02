# import torch
# import torch.nn as nn
# import mydataprocess
# from mydataprocess import dataset, mydataloader
# import option
from torchvision import transforms
# from PIL import Image
# import os
# import matplotlib.pyplot as plt
# import net.model
#
# myoption = option.opt()
# myoption.name = 'color'
# myoption.which_epoch = '190'
# myoption.mode = 'test'
# myoption.load_from_drive = False
# for name,value in vars(myoption).items():
#     print('%s=%s'%(name,value))
#
# mymodel = net.model.single_frame()
# mymodel.initialize(myoption)
#
unloader = transforms.ToPILImage()  # reconvert into PIL image
def imshow(tensor,interval = 0.5):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)
    image = unloader(image)
    image.save('./result/pair_result/1.png')
from utils import fast_check_result
from mydataprocess import mydataloader
import option
import net.model
import os
import time
import torch
myoption = option.opt()
myoption.mode = 'test'
myoption.which_epoch = 40
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

last_frame = fast_check_result.grabdata(myoption,'/home/waihinchan/Desktop/scar/dataset/pair/_0/_0pair0/last_frame.jpg').cuda()
current_frame = fast_check_result.grabdata(myoption,'/home/waihinchan/Desktop/scar/dataset/pair/_0/_0pair0/0.jpg').cuda()
cat_frames = torch.cat([last_frame,current_frame],1)
mymodel = net.model.time_scar()
mymodel.initialize(opt = myoption)
print(mymodel)

z = torch.randn(last_frame.size(0), myoption.z_dim,
                dtype=torch.float32, device=last_frame.get_device())
result = mymodel.netG(z,cat_frames)
imshow(result)