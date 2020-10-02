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

unloader = transforms.ToPILImage()
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
all_images = sorted(all_images)

for image_folder in all_images:
    image_folder_ = sorted(image_folder)
    for i in range(len(image_folder_)):
        if i < len(image_folder_)-1:
            pair_name = os.path.dirname(image_folder_[i])
            label = pair_name.split('/')[-1]
            if not os.path.isdir(os.path.join(pair_name,str(label)+'pair'+str(i))):
                os.mkdir(os.path.join(pair_name,str(label)+'pair'+str(i)))
                shutil.copy(image_folder_[i],os.path.join(pair_name,str(label)+'pair'+str(i)+'/'+str(i)+'.jpg'))
                shutil.copy(image_folder_[i+1], os.path.join(pair_name, str(label)+'pair' + str(i) + '/'+str(i+1) + '.jpg'))
                shutil.copy(image_folder_[-1], os.path.join(pair_name, str(label)+'pair' + str(i) + '/' + 'last_frame' + '.jpg'))
                difference = torch.abs(grabdata(image_folder_[i]) - grabdata(image_folder_[i+1]))
                imsave(difference,dir = os.path.join(pair_name, str(label)+'pair' + str(i) + '/' + 'difference' + '.jpg'))
