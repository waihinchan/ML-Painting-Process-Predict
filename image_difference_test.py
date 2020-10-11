from utils import fast_check_result
import option
from torchvision import transforms
import torch
unloader = transforms.ToPILImage()  # reconvert into PIL image
def imshow(tensor,index = 1):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)
    image = unloader(image)
    image.save('./result/image_difference/' + str(index) +'.jpg')
myoption = option.opt()
last_frame = fast_check_result.grabdata(myoption,'/home/waihinchan/Desktop/scar/dataset/pair/_0/_0pair1/1.jpg').cuda()
current_frame = fast_check_result.grabdata(myoption,'/home/waihinchan/Desktop/scar/dataset/pair/_0/_0pair1/2.jpg').cuda()
difference = torch.abs(current_frame - last_frame)
difference_G = difference[:,0,:,:]
difference_R = difference[:,1,:,:]
difference_B = difference[:,2,:,:]
imshow(difference_G,index='G')
imshow(difference_R,index='R')
imshow(difference_B,index='B')
# 思路
# Let's say we have a last frame, and the current frame
# if we take the last frame and the "real difference" between current frame and the next frame and the current frame as input
# then we input the cat of these 3(current frame ,last frame,difference frame ) to the GAN
# then what the generate will compare with the REAL next frame.

# Then another VAE is to take the last frame and the current frame as input, to generate that "real difference" we need
# then we can random sampling from the normal distribution when testing