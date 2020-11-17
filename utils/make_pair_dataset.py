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
   return [os.path.join(root,folder) for folder in os.listdir(root)]


# TODO: def this from utils fast_check_result in the future
unloader = transforms.ToPILImage()  
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
    for folder in folders: # all images folders
        temp = []
        if not '.DS_Store' in folder:
          for image in os.listdir(folder):
              if is_image_file(image):
                  temp.append(os.path.join(folder,image))
              elif 'seg' in image:
                  segmap_path = os.path.join(folder,image)
          temp = [temp]
          images += temp
    return images


# in the dataset making stage, we don't know how much label CH we have,
# each dataset only has the label they have, so we only save the label

def make_pair_dataset(dataset_dir,step=3):
    all_folder = get_image_floders(dataset_dir)
    all_images = get_all_image(all_folder)
    for image_folder in all_images:
        image_folder.sort() #sort that single image index
        step_ =  step
        granularity_folder = None
        # if not os.path.isdir(folder):
        for i in range(0,len(image_folder),step_):
            pair_name = os.path.dirname(image_folder[i]) # root
            mark = pair_name.split('/')[-1] # which folder
            # pair/00001/grani1/pair0to1 pair0to2

            granularity_folder = os.path.join(pair_name,  'granularity' + str(step_))
            print(granularity_folder)
            if not os.path.isdir(granularity_folder):
              os.mkdir(granularity_folder)

            next_step = step_
            index = i + next_step if i < (len(image_folder)-next_step) else -1
            index_ = index if index is not -1 else len(image_folder)
            folder = os.path.join(granularity_folder,str(mark) + 'pair' + str(i) + 'to' + str(index_))
            # print(folder)
                # the format will be '_00010 pair i to(i+step)'
                # i means the current j means the next
            if not os.path.isdir(folder):
                os.mkdir(folder)
                shutil.copy(image_folder[i],os.path.join(folder,str(mark)+'current.jpg'))
                # current
                shutil.copy(image_folder[index], os.path.join(folder,str(mark)+'next.jpg'))
                # next
                shutil.copy(image_folder[-1], os.path.join(folder,str(mark)+'last.jpg'))
                # last
                difference = torch.abs(grabdata(image_folder[i]) - grabdata(image_folder[index]))
                difference[difference < 0.05] = 0
                imsave(difference,dir = os.path.join(folder,str(mark)+'difference.jpg'))

    print('Done')
def make_pair_dataset_in_granularity(dataset_dir,granularity_list):
  for granularity in granularity_list:
    make_pair_dataset(dataset_dir,granularity)
  
granularity_list = [1,3,5,10,12]
make_pair_dataset('/content/scar/dataset/pair',step=10)

