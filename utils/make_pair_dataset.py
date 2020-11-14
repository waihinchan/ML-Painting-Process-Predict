# import os
# import re
# import shutil
# from PIL import Image
# from torchvision import transforms
# import torch
# IMG_EXTENSIONS = [
#    '.jpg', '.JPG', '.jpeg', '.JPEG',
#    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
# ]
#
# def is_image_file(filename):
#    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
# def get_image_floders(root):
#    return [os.path.join(root,folder) for folder in os.listdir(root)]
# # if os.path.isdir(folder)
#
# # TODO: def this from utils fast_check_result in the future
# unloader = transforms.ToPILImage()  # reconvert into PIL image
# def imsave(tensor,dir):
#     image = unloader(tensor)
#     image.save(dir)
# def grabdata(path):
#     input_image = Image.open(path)
#     pipe = []
#     pipe.append(transforms.ToTensor())
#     pipe = transforms.Compose(pipe)
#     image = pipe(input_image)
#     return image
#
# def get_all_image(folders,use_label=True):
#     images = []
#     for folder in folders: # all images folders
#         temp = []
#         for image in os.listdir(folder):
#             if is_image_file(image):
#                 temp.append(os.path.join(folder,image))
#             elif 'seg' in image:
#                 segmap_path = os.path.join(folder,image)
#         if use_label:
#             assert (segmap_path is not None),'if use label please assign a label_map_folder'
#         temp = [(temp,segmap_path)] if use_label else [temp]
#         images += temp
#     return images
#
#
# # in the dataset making stage, we don't know how much label CH we have,
# # each dataset only has the label they have, so we only save the label
#
# def make_pair_dataset(dataset_dir,total_frame=None,use_label=True,single_CH=True):
#     all_folder = get_image_floders(dataset_dir)
#     all_images = get_all_image(all_folder,use_label)
#
#     if use_label:
#         pipes = []
#         pipes.append(transforms.ToTensor())
#         pipe = transforms.Compose(pipes)
#
#     for _ in all_images:
#         if use_label:
#             image_folder,segmap = _
#             # if use label we have a turple
#             seg_maps_paths = [os.path.join(segmap, img) for img in os.listdir(segmap) if 'label' in img and not 'segmap' in img]
#             seg_maps = [Image.open(img) for img in seg_maps_paths]
#             seg_tensor = [i for i in map(pipe, seg_maps)]
#             for img in os.listdir(segmap):
#                 if 'segmap' in img:
#                     seg_map_path = os.path.join(segmap, img)
#
#         else:
#             image_folder = _
#
#         # image_folder.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1]).group(1)))
#         # TODO: update this sort in the future depend how the dataset format
#         image_folder.sort()
#         step = len(image_folder)//total_frame + 1 if not total_frame == None else 1
#         for i in range(0,len(image_folder),step):
#             if i < (len(image_folder)-step):
#                 pair_name = os.path.dirname(image_folder[i])
#                 mark = pair_name.split('/')[-1]
#                 if not os.path.isdir(os.path.join(pair_name,str(mark)+'pair'+str(i))):
#                     os.mkdir(os.path.join(pair_name,str(mark)+'pair'+str(i)))
#                     # current and next frame
#                     shutil.copy(image_folder[i],os.path.join(pair_name,str(mark)+'pair'+str(i)+'/'+'current'+'.jpg'))
#                     shutil.copy(image_folder[i+step], os.path.join(pair_name, str(mark)+'pair' + str(i) + '/'+'next' + '.jpg'))
#                     # current and next frame
#                     # last frame
#                     # shutil.copy(image_folder[-step], os.path.join(pair_name, str(label)+'pair' + str(i) + '/' + 'last_frame' + '.jpg'))
#                     shutil.copy(image_folder[-1], os.path.join(pair_name, str(mark)+'pair' + str(i) + '/' + 'last_frame' + '.jpg'))
#                     # last frame
#
#                     # difference
#                     difference = torch.abs(grabdata(image_folder[i]) - grabdata(image_folder[i + step]))
#                     if single_CH:
#                         difference = difference[-1,:,:].unsqueeze(0)
#                         difference = torch.cat([difference, difference, difference], dim=0)
#                     # difference[difference<0.1] = 0
#                     imsave(difference,dir = os.path.join(pair_name, str(mark)+'pair' + str(i) + '/' + 'difference'+'.jpg'))
#                     # difference
#                     difference_ = difference[-1,:,:].unsqueeze(0)
#                     if use_label:
#                         shutil.copy(seg_map_path,os.path.join(pair_name, str(mark) + 'pair' + str(i) + '/' + 'segmap' + '.jpg'))
#                         labels = []
#                         count = 0
#                         for label in seg_tensor:
#                             for index in label[difference_ > 0]:
#                                 if not index == 0:
#                                     count+=1
#                                 if count>=5000:
#                                     labels.append(label)
#                                     break
#                         # label = torch.zeros_like(difference)
#                         for k,_ in enumerate(labels,start=0):
#                             imsave(_[-1:, :, :],dir=os.path.join(pair_name, str(mark) + 'pair' + str(i) + '/' + 'label' + str(k) + '.jpg'))
#
#                             # label += _
#                         # label = label[-1:, :, :]
#
#
#
#
#     print('Done')
#
# make_pair_dataset('../dataset/pair',total_frame = 130,single_CH=False,use_label=False )


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
# if os.path.isdir(folder)

# TODO: def this from utils fast_check_result in the future
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

def get_all_image(folders,use_label=None):
    images = []
    for folder in folders: # all images folders
        temp = []
        if not '.DS_Store' in folder:
          for image in os.listdir(folder):
              if is_image_file(image):
                  temp.append(os.path.join(folder,image))
              elif 'seg' in image:
                  segmap_path = os.path.join(folder,image)
          if use_label is not None:
              assert (segmap_path is not None),'if use label please assign a label_map_folder'
          temp = [(temp,segmap_path)] if use_label else [temp]
          images += temp
    return images


# in the dataset making stage, we don't know how much label CH we have,
# each dataset only has the label they have, so we only save the label

def make_pair_dataset(dataset_dir,use_label='wrt_time',granularity = 3,step=3,ttframe=30):
    all_folder = get_image_floders(dataset_dir)
    all_images = get_all_image(all_folder,use_label)

    if use_label:
        pipes = []
        pipes.append(transforms.ToTensor())
        pipe = transforms.Compose(pipes)

    for _ in all_images:
        # this is temp not useful
        if use_label is not None:
            image_folder,segmap = _
            # if use label we have a another folder for the segmap folder
            seg_maps_paths = [os.path.join(segmap, img) for img in os.listdir(segmap) if 'label' in img and not 'segmap' in img]
            seg_maps = [Image.open(img) for img in seg_maps_paths]
            seg_tensor = [i for i in map(pipe, seg_maps)]
            for img in os.listdir(segmap):
                if 'segmap' in img:
                    full_seg_map_path = os.path.join(segmap, img) # just copy the full_segmap/or do this by dataset

        else:
            image_folder = _
        # image_folder.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1]).group(1)))
        # TODO: update this sort in the future depend how the dataset format
        image_folder.sort()
        # step_ = 1 if use_label=='wrt_time' else (len(image_folder)//ttframe+1)
        step_ = 1 if use_label=='wrt_time' else step

        for i in range(0,len(image_folder),step_):
            pair_name = os.path.dirname(image_folder[i])
            mark = pair_name.split('/')[-1]
            if use_label=='wrt_time':
                if i <(len(image_folder)-granularity):
                    for j in range(1,granularity):
                        folder = os.path.join(pair_name, str(mark) + 'pair' + str(i) + 'to' + str(i + j))
                        # the format will be '_00010 pair i to(i+j)'
                        # i means the current j means the next
                        if not os.path.isdir(folder):
                            os.mkdir(folder)
                            shutil.copy(image_folder[i],os.path.join(folder,str(mark)+'current.jpg'))
                            # current
                            shutil.copy(image_folder[i+j], os.path.join(folder,str(mark)+'next.jpg'))
                            # next
                            shutil.copy(image_folder[-1], os.path.join(folder,str(mark)+'last.jpg'))
                            # last
                            difference = torch.abs(grabdata(image_folder[i]) - grabdata(image_folder[i + j]))
                            imsave(difference,dir = os.path.join(folder,str(mark)+'difference.jpg'))
            elif use_label == 'wrt_position':
                next_step = step_
                # print(next_step)
                index = i + next_step if i < (len(image_folder)-next_step) else -1
                index_ = index if index is not -1 else len(image_folder)
                # if i < (len(image_folder)-step):
                folder = os.path.join(pair_name, str(mark) + 'pair' + str(i) + 'to' + str(index_))
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
# for i in range(6,12,3):
#   make_pair_dataset('../dataset/pair',use_label='wrt_position',granularity=4,step=i,ttframe=30)
make_pair_dataset('../dataset/pair',use_label='wrt_position',granularity=4,step=12,ttframe=30)

# if use label + wrt_time , i to j
# if use label + wrt_position, no i to j
