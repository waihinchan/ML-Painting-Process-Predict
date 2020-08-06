import multiprocessing
import torch
cpu_num = multiprocessing.cpu_count()
class opt():
    def __init__(self):
        self.label = True
        # 先这么写着
        self.flip = True
        # make this two are the power of 2
        self.loading_size_w = 1024
        self.inputsize = 512
        self.crop = False
        self.n_downsample_global = 4
        self.mode = "train"
        self.label_nc = 0
        self.checkpoint_dir = "./checkpoint"
        self.name = "./city2"
        self.batchSize = 1
        self.shuffle = True
        self.Nthreads = 4 * cpu_num
        self.gpu_ids = torch.cuda.device_count()