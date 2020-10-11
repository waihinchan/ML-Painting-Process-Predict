import multiprocessing
import torch
import torch.nn as nn
import os
cpu_num = multiprocessing.cpu_count()

class opt():
    def __init__(self):
        self.learningrate = 0.0002
        self.epoch = 400
        self.niter_decay = 300
        self.lsgan = True
        self.mode = "train"
        self.input_chan = 3
        self.output_channel = 3
        self.firstK = 64
        self.checkpoint_dir = "./checkpoint"
        self.load_from_drive = False
        self.name = "pair"
        self.batchSize = 1
        # don't set it to other
        self.shuffle = True
        self.Nthreads = cpu_num / 8  # don't set it too high especially you have a lot cpu....
        self.gpu_ids = torch.cuda.device_count()
        self.which_epoch = '0'
        self.debug = False
        self.num_scale = 3 # this is the multi scale dis
        self.n_blocks = 9 # this is for the resnet block
        self.n_downsample_global = 4 # this is for how much conv
        self.n_past_frames = 1 # this is for how much previous frames cat into the encoder
        self.bs_total_frames = 4 # TBD
        self.input_size = 128
        self.save_result = False
        self.generate_first_frame = True
        self.use_difference = True
        self.norm_type = 'instance'
        # spade's option for generate the style sketch
        # will get rid of it
        self.init_type = 'xavier'
        self.init_variance = 0.02
        self.use_sigmoid = True
        self.upsample_num = 'more' # will change it into num in future
        self.z_dim = 256
        self.use_vae = True
        self.use_spectral = True
        self.norm_G = 'spectralspadebatch3x3'
        self.norm_D = 'spectralbatch'
        self.n_layers_D = 4
        self.num_D = 1
        self.netD_subarch = 'n_layer'
        self.no_ganFeat_loss = True
        # spade's option for generate the style sketch
        # will get rid of it
