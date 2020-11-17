import multiprocessing
import torch
import torch.nn as nn
import os
cpu_num = multiprocessing.cpu_count()

class opt():
    def __init__(self):
        self.learningrate = 0.0002
        self.epoch = 402
        self.niter_decay = 200
        self.lsgan = True
        self.mode = "train"
        self.input_chan = 3
        self.output_channel = 3
        self.firstK = 64
        self.checkpoint_dir = "./checkpoint"
        self.load_from_drive = False
        self.name = "pair"
        self.batchSize = 1
        """
        so far the batchSize is 1, because pytorch has some problems at list or something not tensor if the bs is more than 1
        this can be solved if we found a way make the clean one-hot label from segmap
        """

        self.shuffle = True
        self.Nthreads = cpu_num / 8  # don't set it too high especially you have a lot cpu.... it will cause memory leak in my machine..
        self.gpu_ids = torch.cuda.device_count()
        self.which_epoch = '0'
        self.debug = False
        self.num_scale = 3 # this is the multi scale discriminator, so far we didn't use that
        self.n_blocks = 9 # this is for the resnet block
        self.n_past_frames = 1 # how much n_past_frames will be cat into the discriminator during seq optimization
        self.input_size = 256
        self.save_result = False
        self.upsample_num_ = 5 if self.input_size>=256 else 4
        self.n_downsample_global = 4
        self.use_raw_only = False
        self.generate_first_frame = True # if false this should generate a from another G/or given a
        self.use_difference = True
        self.norm_type = 'instance' # spade's option,will get rid of it
        self.use_label = True # if use semantic segmentation map
        self.label_CH = 10 # this is the one-hot label CH, should match to the classes amount of the label map catalog
        self.use_instance = True # TODO add a optinal in the future
        self.use_wireframe = False # TODO if use the digital painting this can be useful
        self.use_degree = 'wrt_position'
        self.zero_degree = 0.03 # minimum amount of change we don't care
        self.granularity = 5 # granularity of the changing degree
        self.use_restnet = False
        self.l1_lambda = 100
        self.GAN_lambda = 10
        self.Vgg_lambda = 10
        self.Kld_lambda = 0.05
        self.forward = 'pair' # pair or seq, and also seq has two option, see below
        self.z_dim = 256 # for the VAE

        # this 2 is for seq smapling training
        self.smapling_traning = False
        self.max_time_step = 50 # encourage the model finished a painting in X time step








        # # will get rid of it
        # self.init_type = 'xavier'
        # self.init_variance = 0.02
        # self.use_sigmoid = True
        # self.upsample_num = 'more' # will change it into num in future
        #
        # self.use_vae = True
        # self.use_spectral = True
        # self.norm_G = 'spectralspadebatch3x3'
        # self.norm_D = 'spectralbatch'
        # self.n_layers_D = 4
        # self.num_D = 1
        # self.netD_subarch = 'n_layer'
        # self.no_ganFeat_loss = True
        # # spade's option for generate the style sketch
        # # will get rid of it