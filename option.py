import multiprocessing
import torch
import os
cpu_num = multiprocessing.cpu_count()

class opt():
    def __init__(self):
        self.learningrate = 0.0002
        self.epoch = 200
        self.niter_decay = 100
        self.lsgan = True
        self.mode = "train"
        self.input_chan = 3
        self.output_channel = 3
        self.firstK = 64
        self.checkpoint_dir = "./checkpoint"
        self.load_from_drive = False
        self.name = "step"
        self.batchSize = 1
        # don't set it to other
        self.shuffle = True
        self.Nthreads = cpu_num / 8 # don't set it too high especially you have a lot cpu....
        self.gpu_ids = torch.cuda.device_count()
        self.which_epoch = '0'
        self.debug = False
        self.num_scale = 3
        self.n_blocks = 9
        self.n_downsample_global = 4
        # spade
        self.use_sigmoid = True
        self.upsample_num = 4
        self.z_dim = 256
        self.use_spectral = True
        # spade
        self.n_past_frames = 1
        self.bs_total_frames = 4
        self.input_size = 512
        self.shuffle_ecah_time = False
        self.save_result = False
        # this is the last input size
        self.generate_first_frame = True