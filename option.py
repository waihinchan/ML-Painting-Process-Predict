import multiprocessing
import torch
import os
cpu_num = multiprocessing.cpu_count()
class opt():
    def __init__(self):
        # common
        self.learningrate = 0.0002
        self.epoch = 200
        self.niter_decay = 100
        self.CenterCrop = True
        self.crop = False
        # remain to be edit
        self.label = True
        self.flip = True
        self.lsgan = True
        self.inputsize = (256,512)
        # this is the final input size(image) to the model
        self.mode = "train"
        self.input_chan = 3
        self.label_chan = 0
        self.n_frames_G = 3
        self.output_channel = 3
        self.firstK = 64
        self.checkpoint_dir = "./checkpoint"
        self.load_from_drive = False
        # the model root
        self.name = "video"
        # this should be the dataset name and also the model name
        self.batchSize = 1
        self.shuffle = True
        self.Nthreads = 4 * cpu_num
        self.gpu_ids = torch.cuda.device_count()
        self.which_epoch = '0'
        self.debug = False
        self.use_spectral = True
        self.num_scale = 3
        # common
        self.n_blocks = 9
        # pix2pix
        self.n_downsample_global = 4
        self.label_nc = 0
        # not used
        # pix2pix

        # spade
        self.use_sigmoid = True
        self.upsample_num = 4
        self.z_dim = 256
        # spade

        self.n_past_frames = 3

        self.bs_total_frames = 50
        # not sure this will exceed memory...
        self.use_vector = False