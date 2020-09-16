import torch.utils.data

def create_dataset(opt):
    mydataset = None

    from mydataprocess.dataset import video_dataset
    mydataset = video_dataset(opt)
    print("dataset [%s] was created with %s data" % (opt.name,mydataset.__len__()))

    # dataset.initialize(opt)
    # TBD

    return mydataset

class Dataloader():
    def __init__(self,opt):
        self.opt = opt
        self.dataset = create_dataset(self.opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batchSize,
                                                      shuffle=opt.shuffle,
                                                      num_workers=int(opt.Nthreads))


    def load_data(self):
        return self.dataloader

    def __len__(self):

        return self.dataset.__len__()
