import multiprocessing
import torch
import os
cpu_num = multiprocessing.cpu_count()
class opt():
    def __init__(self):
        self.label = True
        self.flip = True
        # make this two are the power of 2
        self.loading_size_w = 1024
        self.inputsize = 512
        self.crop = False
        self.n_downsample_global = 4
        self.mode = "train"
        self.label_nc = 0
        self.input_chan = 3
        self.checkpoint_dir = "./checkpoint"
        # the model root
        self.name = "city2"
        # this should be the dataset name and also the model name
        self.batchSize = 1
        self.shuffle = True
        self.Nthreads = 4 * cpu_num
        self.gpu_ids = torch.cuda.device_count()
        self.which_epoch = '1'
        self.debug = False

