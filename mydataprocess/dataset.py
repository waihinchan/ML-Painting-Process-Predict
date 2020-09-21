import os
import random
import numpy as np
import random
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import re

# *************************data process**************************** #


# **************************data process method**************************
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    the_data = []
    # for the path list
    assert os.path.isdir(dir),'%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                the_data.append(path)

    return the_data


def plotimage(img,interval = 0.5):
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.pause(interval)
    plt.close('all')

def loadimg(path):
    return Image.open(path).convert('RGB')


def how_to_process(opt,img_size):
    w,h = img_size
    input_w  = opt.inputsize if  isinstance(opt.inputsize,int) else opt.inputsize[0]
    input_h = opt.input_w * h // w if isinstance(opt.inputsize,int) else opt.inputsize[-1]

    x = random.randint(0, np.maximum(0, w - opt.input_w))
    y = random.randint(0, np.maximum(0, h - opt.input_h))
    filp = None
    if(opt.flip):
        flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}



def build_pipe(opt):
    """
    :param opt: the target size
    :return: the tensor
    """
    pipes = []
    if opt.CenterCrop:
        pipes.append(transforms.CenterCrop(opt.inputsize))
    else:
        pipes.append(transforms.RandomCrop(opt.inputsize,pad_if_needed=True,padding_mode='edge'))
        # make sure if the orginal image less then the input size
    if opt.flip and 'train' in opt.mode:
        pipes.append(transforms.RandomHorizontalFlip(p=0.5))

    # if opt.resize_or_crop == 'none':
    #     base = float(2 ** opt.n_downsample_global)
    # if opt.netG == 'local':
    #     base *= (2 ** opt.n_local_enhancers)
    # pipes.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    pipes.append(transforms.ToTensor())
    pipes.append(transforms.Normalize((0.5, 0.5, 0.5),
                                          (0.5, 0.5, 0.5)))
    return transforms.Compose(pipes)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


# **************************data process method**************************

"""
video dataset
pipeline -> rescale to the opt.inputimage size -> to tensor
# not sure need to wash the noisy frames(don't know how to do either)
every time get the item, according to the opt.total_frame to return the frame num.
and randomly pick the frame in a Interval
like we want 1000 frames in total (during 1 batchsize train) and 1 data contains of 5000 frames
we will split the dataset into 5000 / 1000 block, pick 1 frame at each block
"""
class step_dataset(data.Dataset):
    def __init__(self,opt):
        super(step_dataset, self).__init__()
        self.opt = opt
        self.data_root_path = os.path.join(os.getcwd(), "dataset")
        print("the root dataset path is " + self.data_root_path)
        self.path = os.path.join(self.data_root_path, opt.name)
        # ../dataset/video/
        print("the dataset path is " + self.path)
        self.dir = sorted([os.path.join(self.path, i) for i in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, i))])
        # ../dataset/video/1
        self.opt.bs_total_frames = min(self.opt.bs_total_frames,300)
        # protect the machine XD
    def __getitem__(self, index):
        path = self.dir[index]
        # ../dataset/video/index
        all_frames_path = [i for i in os.listdir(path) if is_image_file(i)]
        all_frames_path.sort(key=lambda x: int(re.match('(\d+)\.', x).group(1)))
        all_frames_path = [os.path.join(path, i) for i in all_frames_path]
        # all the full path of each frames are include inside
        frames = []

        # hard coding
        for i in range(0,3):
            frames.append(all_frames_path[i])
        for j in range(1,3):
            frames.append(all_frames_path[-(3-i)])

        frames = [Image.open(frame) for frame in frames]
        pipes = []
        pipes.append(transforms.Resize(self.opt.input_size))
        pipes.append(transforms.ToTensor())
        # pipes.append(transforms.Normalize((0.5, 0.5, 0.5),
        #                                   (0.5, 0.5, 0.5)))


        pipe = transforms.Compose(pipes)
        tensor_list = [i for i in map(pipe,frames)]

        last_frame = pipe(Image.open(all_frames_path[-1]))
        # not sure the sequence, need a test...
        return {'frames':tensor_list,'target':last_frame}


    def __len__(self):
        return len(self.dir)//self.opt.batchSize * self.opt.batchSize
        # remain to test

class video_dataset(data.Dataset):
    def __init__(self,opt):
        super(video_dataset, self).__init__()
        self.opt = opt
        self.data_root_path = os.path.join(os.getcwd(), "dataset")
        print("the root dataset path is " + self.data_root_path)
        self.path = os.path.join(self.data_root_path, opt.name)
        # ../dataset/video/
        print("the dataset path is " + self.path)
        self.dir = sorted([os.path.join(self.path, i) for i in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, i))])
        # ../dataset/video/1
        self.opt.bs_total_frames = min(self.opt.bs_total_frames,300)
        # protect the machine XD
    def __getitem__(self, index):
        path = self.dir[index]
        # ../dataset/video/index
        all_frames_path = [i for i in os.listdir(path) if is_image_file(i)]
        all_frames_path.sort(key=lambda x: int(re.match('(\d+)\.', x).group(1)))
        all_frames_path = [os.path.join(path, i) for i in all_frames_path]

        # all the full path of each frames are include inside

        # randomly pick the frame
        frames = []
        if self.opt.bs_total_frames>=len(all_frames_path):
            mult = int(np.ceil(self.opt.bs_total_frames / len(all_frames_path)))
            # remain to test
            for _ in range(mult):
                all_frames_path += all_frames_path
            frames = sorted(all_frames_path)

        block = len(all_frames_path)//self.opt.bs_total_frames
        for i in range(0,len(all_frames_path),block):
            # if self.opt.shuffle_each_time:
            #     pick_index = min(i+random.randint(0,block),len(all_frames_path)-1)
            # else:
            pick_index = min(i+block//2,len(all_frames_path)-1)
            # pick each block middle or first or last?
            frames.append(all_frames_path[pick_index])
        # randomly pick the frame
        # build the pre-process pipie line
        # as i already resize the frames. so only need a to tensor
        frames = [Image.open(frame) for frame in frames]
        pipes = []
        pipes.append(transforms.Resize(self.opt.input_size))

        pipes.append(transforms.ToTensor())
        pipes.append(transforms.Normalize((0.5, 0.5, 0.5),
                                          (0.5, 0.5, 0.5)))


        pipe = transforms.Compose(pipes)
        tensor_list = [i for i in map(pipe,frames)]

        last_frame = pipe(Image.open(all_frames_path[-1]))
        # not sure the sequence, need a test...
        return {'frames':tensor_list,'target':last_frame}


    def __len__(self):
        return len(self.dir)//self.opt.batchSize * self.opt.batchSize
        # remain to test


# class dataset_070(data.Dataset):
#     # remain some error handle need to update
#     # remain step params to add (decide how many step would take)
#     def  __init__(self,opt):
#         super(dataset_070, self).__init__()
#         self.opt = opt
#         self.data_root_path = os.path.join(os.getcwd(), "dataset")
#
#         print("the root path is " + self.data_root_path)
#         self.path = os.path.join(self.data_root_path, opt.name)
#         self.dir = sorted(os.listdir(self.path))
#         # this willreturn the full path of each image
#         print("the dataset path is " + self.path)
#
#     def __getitem__(self, index):
#
#         """
#         :param index: index should be NNN 00N 001 002 003 004
#         :return: return a dist of multi inputs
#         """
#         index = str(index)
#         # pass through the
#         real_index =index.rjust(3,'0')
#         # get the format 00N
#         pathlist = sorted([i for i in self.dir if i.startswith(real_index)])
#         # str -> path
#         rawdatalist = [Image.open(os.path.join(self.path,i)) for i in pathlist]
#         # raw PIL image
#
#         # params = how_to_deal(self.opt,rawdatalist[-1].size)
#         # transforms_pipe = build_pipe(self.opt,params, method=Image.NEAREST, normalize=False)
#         transforms_pipe = build_pipe(self.opt)
#         datalist = [i for i in map(transforms_pipe,rawdatalist)]
#
#         # this should be tensor
#         # remain to test whether need * 255.0
#
#         return {'step_1':datalist[0],'step_2':datalist[1],'target':datalist[-1]}
#
#     def __len__(self):
#         last = self.dir[-1]
#         return int(last[:3])//self.opt.batchSize * self.opt.batchSize + 1
#         # return int(last[:3])






