import os
import random
import numpy as np

import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


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



"""
思路：
1.有一个放大的size
2.有一个最终的inputsize
3.先找到你的原图多大，同时找到随机裁剪点
4.然后把原图放大/或缩小到指定尺寸
5.然后找到随机裁剪点进行裁剪

问题在于 如果inputsize 不是1:1 就会很麻烦
o 1024 512
t 512 512

512/512 = 1024/n
n = 1024
oh -> 1024

"""


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

# if one side of the orginal image is less than the inputsize, the crop position would be 0.
# so in the scale function need the resize the image (especially when one side is less than the input)
# def howtotest(inputsize,img_size):
#     w,h = img_size
#     input_w  = inputsize if  isinstance(inputsize,int) else inputsize[0]
#
#     input_h = input_w * h // w if isinstance(inputsize,int) else inputsize[1]
#
#     x = random.randint(0, np.maximum(0, w - input_w))
#     y = random.randint(0, np.maximum(0, h - input_h))
#     filp = None
#
#     flip = random.random() > 0.5
#     return {'crop_pos': (x, y), 'flip': flip}
#
# a = (256,1024)
# print(howtotest((512,2048),a))


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



class dataset_070(data.Dataset):
    # remain some error handle need to update
    # remain step params to add (decide how many step would take)
    def  __init__(self,opt):
        super(dataset_070, self).__init__()
        self.opt = opt
        self.data_root_path = os.path.join(os.getcwd(), "dataset")

        print("the root path is " + self.data_root_path)
        self.path = os.path.join(self.data_root_path, opt.name)
        self.dir = sorted(os.listdir(self.path))
        # this willreturn the full path of each image
        print("the dataset path is " + self.path)

    def __getitem__(self, index):

        """
        :param index: index should be NNN 00N 001 002 003 004
        :return: return a dist of multi inputs
        """
        index = str(index)
        # pass through the
        real_index =index.rjust(3,'0')
        # get the format 00N
        pathlist = sorted([i for i in self.dir if i.startswith(real_index)])
        # str -> path
        rawdatalist = [Image.open(os.path.join(self.path,i)) for i in pathlist]
        # raw PIL image

        # params = how_to_deal(self.opt,rawdatalist[-1].size)
        # transforms_pipe = build_pipe(self.opt,params, method=Image.NEAREST, normalize=False)
        transforms_pipe = build_pipe(self.opt)
        datalist = [i for i in map(transforms_pipe,rawdatalist)]

        # this should be tensor
        # remain to test whether need * 255.0

        return {'step_1':datalist[0],'step_2':datalist[1],'target':datalist[-1]}

    def __len__(self):
        last = self.dir[-1]
        return int(last[:3])//self.opt.batchSize * self.opt.batchSize + 1
        # return int(last[:3])


class facades(data.Dataset):
    # this is for the images which are combine together, and in train test eval etc folder
    # but still some details reamin to update
    def __init__(self,opt):
        super(facades, self).__init__()
        self.opt = opt
        self.data_root_path = os.path.join(os.getcwd(), "dataset")

        print("the root path is " + self.data_root_path)

        self.path = os.path.join(self.data_root_path, opt.name)
        self.path = self.path + '/train'
        # ./dataset/facades/train
        self.dir = [i for i in os.listdir(self.path) if i.endswith('.jpg') ]
        # need update here

        # this willreturn the full path of each image
        print("the dataset path is " + self.path)

    def __getitem__(self, index):
        index = str(index+1)
        rawimage = Image.open(self.path + '/' + index + '.jpg')
        w,h = rawimage.size
        pipe = []
        pipe.append(transforms.ToTensor())
        transform_pipe = transforms.Compose(pipe)
        rawtensor = transform_pipe(rawimage)
        return {'image':rawtensor[:,:,:int(w/2)],'label':rawtensor[:,:,int(w/2):]}


    def __len__(self):
        return len(self.dir) // self.opt.batchSize * self.opt.batchSize



#
# for i in os.listdir('/Users/waihinchan/Documents/mymodel/scar/dataset/070'):
#     print(i)

# print(sorted(os.listdir('/Users/waihinchan/Documents/mymodel/scar/dataset/070')))
