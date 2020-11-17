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

def __clean_noise(tensor):
    tensor[tensor<0.5] = 0
    tensor[tensor >= 0.5] = 1
    return tensor

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

class seq_dataset(data.Dataset):
    def __init__(self,opt):
        super(seq_dataset, self).__init__()
        self.opt = opt
        self.data_root_path = os.path.join(os.getcwd(), "dataset")
        print("the root path is " + self.data_root_path)
        self.path = os.path.join(self.data_root_path, opt.name)
        print("the dataset path is " + self.path)
        all = os.walk(self.path)
        # this for storage all the video folder (each folder has many frames)
        self.all_seq = [os.path.join(self.path,video) for video in os.listdir(self.path) if os.path.isdir(os.path.join(self.path,video))]

        # we will iter and sort the sub folder when __getitem__
        self.all_seq.sort()
    def get_one_pairs(self,path):# this will return a single dict, same like the pair dataset

        return_list = {'current': None, 'last': None,'next':None}
        # ************************* this is the fixed stuff need to be return ************************* #
        pair_paths = [os.path.join(path, img) for img in os.listdir(path) if is_image_file(img)]        # get all the img
        pair_paths = sorted(pair_paths)
        frames = [Image.open(img) for img in pair_paths if is_image_file(img) and not 'label' in img and not 'single' in img]
        # ************************* all the pair folder ************************* #
        segmap_folder = os.path.dirname(os.path.dirname(pair_paths[0])) + '/segmap'  # not sure if this still working

        # ************************* the segmaps folder ************************* #

        if self.opt.use_label:
            segmap_path = [Image.open(img) for img in pair_paths if is_image_file(img) and 'single' in img]
            # in case of if we specify a certain label map, generally are different parts of the full segmap
            full_segmap = [Image.open(os.path.join(segmap_folder, img)) for img in os.listdir(segmap_folder) if is_image_file(img) and not 'singlesegmap' in img]
            # if didn't specify a certain label map. the dataset will return the full segmap
        # ************************* get the label map ************************* #

        if self.opt.use_degree == 'wrt_position':
            assert len(frames)>=4,'if use degree base on position, please specify a difference image,the current frames dataset is less than 3 (current and last frames are necessary input)'
            single_labels = [Image.open(os.path.join(segmap_folder, img)) for img in os.listdir(segmap_folder) if is_image_file(img) if is_image_file(img) and 'label' in img and not 'segmap' in img ]
        # ************************* get the one-hot label map ************************* #

        pipes = []
        pipes.append(transforms.Resize(self.opt.input_size))
        pipes.append(transforms.ToTensor())
        # pipes.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        # not sure should use this... if use this the difference will dissappear..
        pipe = transforms.Compose(pipes)
        # ***************************  transform pipes ***************************

        frames_list = [i for i in map(pipe,frames)]
        return_list['current'] = frames_list[0]
        return_list['next'] = frames_list[-1]
        return_list['last'] = frames_list[-2]
        if self.opt.use_difference:
            return_list['difference'] = frames_list[1]
        # if use wireframe, generally it's generated from another generator..,no given by the dataset
        if self.opt.use_label:
            if len(segmap_path) == 0:
                segmap = pipe(full_segmap[0])
            else:
                segmap = self.get_segmap(segmap_path,pipe)
            return_list['label'] = segmap
        if self.opt.use_degree == 'wrt_position':
            label_pipes = []
            label_pipes.append(transforms.ToTensor())
            label_pipes.append(transforms.Lambda(lambda img: __clean_noise(img)))
            label_pipes.append(transforms.Resize(self.opt.input_size))
            label_pipe = transforms.Compose(pipes)
            single_label_list = [j for j in map(label_pipe, single_labels)]
            return_list['segmaps'] = single_label_list # this is list for caculate the degree
        # *************************** make the return list ***************************

        return return_list
    def __getitem__(self, index):# this will return a list consist of many dict
        video = self.all_seq[index]
        all_pairs = [os.path.join(video,pair) for pair in os.listdir(video) if 'pair' in pair]
        # all_pairs.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1].split('pair')[-1]).group(1)))
        all_pairs.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1].split('pair')[-1].split('to')[0]).group(1)))

        # dataset/pair/00001/_00010pair159
        # split the / and take _00010pair159
        # split the pair and take the 159
        return [self.get_one_pairs(single_pair) for single_pair in all_pairs]
    def get_segmap(self,index_list,pipe):
        index_tensor_list = [i for i in map(pipe,index_list)]
        segmap = torch.cat(index_tensor_list)
        segmap_ = torch.sum(segmap,0,keepdim=True)
        return segmap_
    def __len__(self):
        return len(self.all_seq)//self.opt.batchSize * self.opt.batchSize


class pair_dataset(data.Dataset):
    def __init__(self,opt):
        super(pair_dataset, self).__init__()
        self.opt = opt
        self.data_root_path = os.path.join(os.getcwd(), "dataset")
        print("the root path is " + self.data_root_path)
        self.path = os.path.join(self.data_root_path, opt.name)
        print("the dataset path is " + self.path)
        all = os.walk(self.path)
        all_pairs = []
        for root, dirs, files in all:
            for dir in dirs:
                if 'pair' in dir:
                    all_pairs.append(os.path.join(root, dir))
        self.all_pairs = all_pairs
        print(self.all_pairs)
        self.all_pairs.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1].split('pair')[-1]).group(1)))


    def __getitem__(self, index):
        return_list = {'current': None, 'last': None,'next':None}
        # ************************* this is the fixed stuff need to be return ************************* #

        pair_paths = [os.path.join(self.all_pairs[index],img) for img in os.listdir(self.all_pairs[index])]
        pair_paths = sorted(pair_paths)
        frames = [Image.open(img) for img in pair_paths if is_image_file(img) and not 'label' in img and not 'single' in img]
        # ************************* all the pair folder ************************* #
        parent = os.path.dirname(os.path.dirname(os.path.dirname(pair_paths[0])))
        segmap_folder = parent + '/segmap' if self.opt.use_label else None # TODO remake the dataset name in the future
        # ************************* the segmaps folder ************************* #

        if self.opt.use_label:
            segmap_path = [Image.open(img) for img in pair_paths if is_image_file(img) and 'single' in img]
            # in case of if we specify a certain label map, generally are different parts of the full segmap
            full_segmap = [Image.open(os.path.join(segmap_folder, img)) for img in os.listdir(segmap_folder) if is_image_file(img) and not 'singlesegmap' in img]
            # if didn't specify a certain label map. the dataset will return the full segmap
        # ************************* get the label map ************************* #

        if self.opt.use_degree == 'wrt_position':
            assert len(frames)>=4,'if use degree base on position, please specify a difference image,the current frames dataset is less than 3 (current and last frames are necessary input)'
            single_labels = [Image.open(os.path.join(segmap_folder, img)) for img in os.listdir(segmap_folder) if is_image_file(img) if is_image_file(img) and 'label' in img and not 'segmap' in img ]
        # ************************* get the one-hot label map ************************* #

        pipes = []
        pipes.append(transforms.Resize(self.opt.input_size))
        pipes.append(transforms.ToTensor())
        # pipes.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        # not sure should use this... if use this the difference will dissappear..
        pipe = transforms.Compose(pipes)
        # ***************************  transform pipes ***************************

        frames_list = [i for i in map(pipe,frames)]
        return_list['current'] = frames_list[0]
        return_list['next'] = frames_list[-1]
        return_list['last'] = frames_list[-2]
        if self.opt.use_difference:
            return_list['difference'] = frames_list[1]
        # if use wireframe, generally it's generated from another generator..,no given by the dataset
        if self.opt.use_label:
            if len(segmap_path) == 0:
                segmap = pipe(full_segmap[0])
            else:
                segmap = self.get_segmap(segmap_path,pipe)
            return_list['label'] = segmap
        if self.opt.use_degree == 'wrt_position':
            label_pipes = []
            label_pipes.append(transforms.ToTensor())
            label_pipes.append(transforms.Lambda(lambda img: __clean_noise(img)))
            label_pipes.append(transforms.Resize(self.opt.input_size))
            label_pipe = transforms.Compose(pipes)
            single_label_list = [j for j in map(label_pipe, single_labels)]
            return_list['segmaps'] = single_label_list # this is list for caculate the degree
        # *************************** make the return list ***************************

        return return_list
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






