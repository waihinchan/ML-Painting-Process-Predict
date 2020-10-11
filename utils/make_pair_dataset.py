import os
import re
import shutil
from PIL import Image
from torchvision import transforms
import torch
IMG_EXTENSIONS = [
   '.jpg', '.JPG', '.jpeg', '.JPEG',
   '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
   return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def get_image_floders(root):
   return [os.path.join(root,folder) for folder in os.listdir(root) if folder !='.DS_Store']

unloader = transforms.ToPILImage()  # reconvert into PIL image
def imsave(tensor,dir):
    image = unloader(tensor)
    image.save(dir)
def grabdata(path):
    input_image = Image.open(path)
    pipe = []
    pipe.append(transforms.ToTensor())
    pipe = transforms.Compose(pipe)
    image = pipe(input_image)
    return image

def get_all_image(folders):
   images = []
   for folder in folders:

       temp = [[os.path.join(folder,image) for image in os.listdir(folder) if is_image_file(image)]]
       images += temp
   return images

all_folder = get_image_floders('../dataset/pair')
all_images = get_all_image(all_folder)
all_images
step = 3
import torch.nn as nn
# threshold = 0.2
# TH = nn.Threshold(threshold, 1)
l1loss = nn.L1Loss()
import torch.nn as nn

threshold = torch.tensor([0.0])
total = 0
count = 0
for image_folder in all_images:
    image_folder.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1]).group(1)))
    for i in range(0,len(image_folder),step):
        if i < (len(image_folder)-step):
            pair_name = os.path.dirname(image_folder[i])
            label = pair_name.split('/')[-1]
            if not os.path.isdir(os.path.join(pair_name,str(label)+'pair'+str(i))):

                # if l1loss(difference, torch.ones_like(difference))*100000 >= 99356:
                    os.mkdir(os.path.join(pair_name,str(label)+'pair'+str(i)))
                    shutil.copy(image_folder[i],os.path.join(pair_name,str(label)+'pair'+str(i)+'/'+str(i)+'.jpg'))
                    shutil.copy(image_folder[i+step], os.path.join(pair_name, str(label)+'pair' + str(i) + '/'+str(i+step) + '.jpg'))
                    # shutil.copy(image_folder[-1], os.path.join(pair_name, str(label)+'pair' + str(i) + '/' + 'segmap' + '.jpg'))
                    shutil.copy(image_folder[-step], os.path.join(pair_name, str(label)+'pair' + str(i) + '/' + 'last_frame' + '.jpg'))
                    difference = torch.abs(grabdata(image_folder[i]) - grabdata(image_folder[i + step]))
                    # difference = torch.abs(grabdata(image_folder[i]) - grabdata(image_folder[i+step]))[-1,:,:].unsqueeze(0)
                    # difference = torch.where(difference > 0.1, difference, threshold)
                    # difference = torch.cat([difference,difference,difference],dim=0)
                    imsave(difference,dir = os.path.join(pair_name, str(label)+'pair' + str(i) + '/' + 'difference' + str(i) + '.jpg'))
