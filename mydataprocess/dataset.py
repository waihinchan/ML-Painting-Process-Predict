import os
import random
import numpy as np
import random
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import re
import torch

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


class single_image(data.Dataset):
    def __init__(self,opt):
        super(single_image, self).__init__()
        self.opt = opt
        self.data_root_path = os.path.join(os.getcwd(), "dataset")
        print("the root dataset path is " + self.data_root_path)
        self.path = os.path.join(self.data_root_path, opt.name)
        # ../dataset/video/
        print("the dataset path is " + self.path)
        self.all_images = sorted([os.path.join(self.path, i) for i in os.listdir(self.path) if is_image_file(os.path.join(self.path, i))])
        # ../dataset/video/1
    def __getitem__(self, index):
        tensor = Image.open(self.all_images[index])
        pipes = []
        pipes.append(transforms.Resize(self.opt.input_size))
        pipes.append(transforms.CenterCrop(self.opt.input_size))
        pipes.append(transforms.ToTensor())
        pipe = transforms.Compose(pipes)
        pipes.append(transforms.Normalize((0.5, 0.5, 0.5),
                                          (0.5, 0.5, 0.5)))
        head = pipe(tensor)
        return head


    def __len__(self):
        return len(self.all_images)//self.opt.batchSize * self.opt.batchSize
        # remain to test

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
        for i in range(1,3):
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

class pair_dataset(data.Dataset):
    def __init__(self,opt):
        super(pair_dataset, self).__init__()
        self.opt = opt
        self.data_root_path = os.path.join(os.getcwd(), "dataset")

        print("the root path is " + self.data_root_path)

        self.path = os.path.join(self.data_root_path, opt.name)
        # ./dataset/datasetname/
        # this willreturn the full path of each image
        print("the dataset path is " + self.path)
        all = os.walk(self.path)
        all_pairs = []
        for root, dirs, files in all:
            for dir in dirs:
                if 'pair' in dir:
                    all_pairs.append(os.path.join(root, dir))
        self.all_pairs = sorted(all_pairs)


    def __getitem__(self, index):
        pair_paths = [os.path.join(self.all_pairs[index],img) for img in os.listdir(self.all_pairs[index])]
        pair_paths = sorted(pair_paths)
        frames = [Image.open(img) for img in pair_paths]
        pipes = []
        pipes.append(transforms.Resize(self.opt.input_size))
        # pipes.append(transforms.CenterCrop(self.opt.input_size))
        pipes.append(transforms.ToTensor())
        # pipes.append(transforms.Normalize((0.5, 0.5, 0.5),
        #                                   (0.5, 0.5, 0.5)))
        # not sure should use this... if use this the difference will dissappear.. but should be in the data not the tensor
        pipe = transforms.Compose(pipes)
        tensor_list = [i for i in map(pipe,frames)]
        # TODO: will get rid of this in the future
        label = tensor_list[-2][0:3,:,:]
        # label[label==0] = 1
        label[label > 0] = 1
        label = label[-1:,:,:]
        return {'current':tensor_list[0],'next':tensor_list[1],'last':tensor_list[-1],'difference':tensor_list[2],'label':label}
        # return {'current':tensor_list[0],'last':tensor_list[-1],'difference':tensor_list[2],'next':tensor_list[1]}



    def __len__(self):
        return len(self.all_pairs) // self.opt.batchSize * self.opt.batchSize

# this is for the color to sketch dataset
class colordataset(data.Dataset):
    def __init__(self,opt):
        super(colordataset, self).__init__()
        self.opt = opt
        self.data_root_path = os.path.join(os.getcwd(), "dataset")

        print("the root path is " + self.data_root_path)

        self.path = os.path.join(self.data_root_path, opt.name)
        self.path = self.path + '/train'
        # ./dataset/datasetname/train
        self.dir = sorted([i for i in os.listdir(self.path) if is_image_file(i)])

        print("the dataset path is " + self.path)

    def __getitem__(self, index):
        rawimage = Image.open(os.path.join(self.path , self.dir[index]))
        w,h = rawimage.size
        pipe = []
        # pipe.append(transforms.Resize(self.opt.input_size))
        pipe.append(transforms.ToTensor())
        # pipe.append(transforms.Normalize((0.5, 0.5, 0.5),
        #                                   (0.5, 0.5, 0.5)))

        transform_pipe = transforms.Compose(pipe)
        rawtensor = transform_pipe(rawimage)
        return {'input':rawtensor[:,:,:int(w/2)],'target':rawtensor[:,:,int(w/2):]}

    def __len__(self):
        return len(self.dir) // self.opt.batchSize * self.opt.batchSize
# this is for the color to sketch dataset





