from PIL import Image
from torchvision import transforms
import torch
unloader = transforms.ToPILImage()
def imsave(tensor,index,dir = './result/video_result/'):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    # image = torch.clamp(image,0,1)
    image = unloader(image)
    image.save( dir+ str(index) + '.jpg')
import numpy as np
def grabdata(path,opt=None):
    input_size = 256 if opt == None else opt.input_size
    input_image = Image.open(path)
    pipe = []
    pipe.append(transforms.Resize(input_size))
    pipe.append(transforms.ToTensor())
    pipe = transforms.Compose(pipe)
    image = pipe(input_image)
    return image.unsqueeze(0)