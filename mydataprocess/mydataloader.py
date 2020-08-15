import torch.utils.data

def create_dataset(opt):
    mydataset = None

    from mydataprocess.dataset import dataset_070
    from mydataprocess.dataset import commondataset
    # mydataset = dataset_070(opt)
    mydataset = commondataset(opt)
    print("dataset [%s] was created" % (opt.name))

    # dataset.initialize(opt)
    # TBD

    return mydataset

class Dataloader():
    def __init__(self,opt):
        self.opt = opt
        # 先把参数存起来
        self.dataset = create_dataset(self.opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batchSize,
                                                      shuffle=opt.shuffle,
                                                      num_workers=int(opt.Nthreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):

        return self.dataset.__len__()
        # TBD