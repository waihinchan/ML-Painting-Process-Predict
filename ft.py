import torch
import option
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Loss import gram
import time
# from utils import visual
import option
# import generator
# myopt = option.opt
import os
import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt



# this is for test the dataset working
from mydataprocess import mydataloader
myoption = option.opt()
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

dataloader = mydataloader.Dataloader(myoption)

thedataset = dataloader.load_data()

for data in thedataset:
    frames = data['frames']
    last_frames = data['last_frame']
    print('******************** printing frames shape ********************')
    print(frames[-1].shape)
    print('******************** printing last frame shape ********************')
    print(last_frames.shape)
# this is for test the dataset working