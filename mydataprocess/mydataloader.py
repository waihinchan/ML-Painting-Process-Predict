import torch.utils.data

def create_dataset(opt):
    mydataset = None
    if "video" in opt.name:
        from mydataprocess.dataset import video_dataset
        mydataset = video_dataset(opt)
    elif "step" in opt.name:
        from mydataprocess.dataset import step_dataset
        mydataset = step_dataset(opt)
    elif "color" in opt.name:
        from mydataprocess.dataset import colordataset
        mydataset = colordataset(opt)
    elif 'pair' in opt.name:
        from mydataprocess.dataset import pair_dataset
        mydataset = pair_dataset(opt)
    elif 'single' in opt.name:
        from mydataprocess.dataset import single_image
        mydataset = single_image(opt)

    print("dataset [%s] was created with %s data" % (opt.name,mydataset.__len__()))
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
