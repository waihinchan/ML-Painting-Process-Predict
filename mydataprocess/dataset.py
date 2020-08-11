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


def how_to_deal(opt,img_size):

    """
    # not sure can i flip if it would effect the result or improve it
    # this is for how to deal with the image, by passing the requirement and the image size itself
    # target -> given a final size, rescale or cut the image to that size, maybe using crop or rescale or resize
    # the pix2pixHD using the same w and h if the option is resize
    # i will use rescale instead (don't know why resize mean w and h be the same)

    :param opt: using opt.input_size , opt.loading_size_w ,  opt.flip
    :param img_size: img_size is a tuple (from PIL image.size())
    :return: where a dict to tell where to crop , flip or not
    """

    w,h = img_size
    loading_w  = opt.loading_size_w
    loading_h = opt.loading_size_w * h // w

    # intput_w,input_h = opt.inputsize
    x = random.randint(0, np.maximum(0, loading_w - opt.inputsize))
    y = random.randint(0, np.maximum(0, loading_h - opt.inputsize))

    filp = None
    if(opt.flip):
        flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def __crop(img, pos, size):
    """
    :param img: the PIL IMAGE
    :param pos: where to cut , pass the "how_to_deal" 's return in it
    :param size: target size
    :return: cropped image
    """
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))

    return img


def __scale_width(img, target_width, method=Image.BICUBIC):
    """
    :param img: the orginal PIL image
    :param target_width: given by the option opt.loading_size_w
    :param method: default BICUBIC remain to test
    :return: the rescaled image

    """

    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh // ow)
    return img.resize((w, h), method)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)
    # https://datascience.stackexchange.com/questions/20179/what-is-the-advantage-of-keeping-batch-size-a-power-of-2

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def build_pipe(opt,params,method = Image.BICUBIC, normalize=True):
    """
    crop will pick a random position then crop to the same W and H size

    :param opt:
    :param params:
    :param method:
    :param normalize:
    :return: the pipe (deal with the img then turn it into tensor)
    """
    pipe = []
    # this is for restore the a series of the operation to the image

    # no resize option here, if the orginal size not match to the target size
    # using lambada resize, else return the orginal img
    pipe.append(transforms.Lambda(lambda img: __scale_width(img, opt.loading_size_w, method)))
    if opt.crop:
        pipe.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.inputsize)))

    else:
        base = float(2 ** opt.n_downsample_global)
        # according to the num of the downsample to make the power of 2
        # can imprve the performance when running at GPU
        # don't know why should be the num of the n_downsample_global
        pipe.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.mode == "train" and opt.flip:
        pipe.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    pipe.append(transforms.ToTensor())
    if normalize:
        pipe.append(transforms.Normalize((0.5, 0.5, 0.5),
                                         (0.5, 0.5, 0.5)))

    return transforms.Compose(pipe)

# **************************data process method**************************

# .dataset/070/001-a 001-b 001-c 002-a 002-b 002-c
# target
# 1. make a list
# 2. list[-1] = target image
# 3. list[0] = 1step
# 4. one params = pick one - don't know yet
# 5. for now only pick 0 -> -1

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

        params = how_to_deal(self.opt,rawdatalist[-1].size)
        transforms_pipe = build_pipe(self.opt,params, method=Image.NEAREST, normalize=False)
        datalist = [i*255.0 for i in map(transforms_pipe,rawdatalist)]

        # this should be tensor
        # remain to test whether need * 255.0

        return {'step_1':datalist[0],'step_2':datalist[1],'target':datalist[-1]}

    def __len__(self):
        last = self.dir[-1]
        return int(last[:3])//self.opt.batchSize * self.opt.batchSize + 1
        # return int(last[:3])


class myDataset(data.Dataset):
    def  __init__(self,opt):
        """
        :param opt:
        using opt.name as the path
        when indicate where is the dataset,
        put it under the the dataset folder and pass the folder name as the param
        """

        super(myDataset, self).__init__()
        self.opt = opt
        # ********************** for the root path ********************** #

        self.data_root_path = os.path.join(os.getcwd(), "dataset")
        # find the root path
        print("the root path is " + self.data_root_path)
        self.path = os.path.join(self.data_root_path, opt.name)
        print("the dataset path is " + self.path)
        # join the exact folder

        # ********************** for the root path ********************** #

        # ********************** for the input image ********************** #

        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        # this is the input folder and the dataset
        self.dir_A = os.path.join(self.path, self.opt.mode + dir_A)
        # opt.mode = opt.phase / provide train or test
        # the whole path would like rootpath/dataset/yourdatasetname/train_A
        self.A_paths = sorted(make_dataset(self.dir_A))
        # this is all the image path inside the dir_A

        # ********************** for the input image ********************** #

        # ********************** for the target image ********************** #

        # so far i only rebuild the input label_image to the real image
        # so i think i don't need the label channel(maybe)
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(self.path, self.opt.mode + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        # ********************** for the target image ********************** #

        # not sure how instant and feature map working
        # TBD


    def __getitem__(self, index):

        """

        :param index: which input(the number of the paired input)
        :return: return a dist of multi inputs

        """
        # img = Image.open(self.image_file[index])

        A = Image.open(self.A_paths[index])
        # this is the PIL image format
        params = how_to_deal(self.opt,A.size)
        transforms_pipe_A = build_pipe(self.opt,params, method=Image.NEAREST, normalize=False)
        A_tensor = transforms_pipe_A(A)*255.0
        # let's say we don't use label channel input

        B = Image.open(self.B_paths[index])
        transforms_pipe_B = build_pipe(self.opt,params)
        B_tensor = transforms_pipe_B(B)

        input_dict = {
            "label": A_tensor,
            "image": B_tensor
        }

        return input_dict

    def __len__(self):

        # return len(self.A_paths)//self.opt.batchSize * self.opt.batchSize
        # don't know why do this
        # let's test
        return len(self.A_paths)


#
# for i in os.listdir('/Users/waihinchan/Documents/mymodel/scar/dataset/070'):
#     print(i)

# print(sorted(os.listdir('/Users/waihinchan/Documents/mymodel/scar/dataset/070')))
