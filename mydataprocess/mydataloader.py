import torch.utils.data

def create_dataset(opt):
    mydataset = None
    from mydataprocess.dataset import myDataset

    mydataset = myDataset(opt)
    # 传递的opt中包含路径
    # 所以思路是先建立好文件夹
    # 然后这里就直接给文件夹路径/名字
    # 还要包含train_A train_B 之类的

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

        return len(self.dataset)
        # return min(len(self.dataset), self.opt.max_dataset_size)
        # TBD