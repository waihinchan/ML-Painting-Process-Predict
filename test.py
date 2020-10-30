from mydataprocess import mydataloader
import option
import net.model
import os
from utils.fast_check_result import imsave
myoption = option.opt()
myoption.batchSize = 1
myoption.input_size = 256
myoption.name = 'pair'
myoption.mode = 'test'
myoption.use_label= True
myoption.shuffle = False
myoption.which_epoch = 400
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))
dataloader = mydataloader.Dataloader(myoption)
pair_data = dataloader.load_data()
mymodel = net.model.label_VAE()
mymodel.initialize(opt = myoption)
import torchvision.transforms as transforms
import PIL.Image as Image
import re
import torch
fwmode = 'seq'
pipes = []
pipes.append(transforms.Resize(myoption.input_size))
pipes.append(transforms.ToTensor())
pipe = transforms.Compose(pipes)
def get_tensor(path):
   return pipe(Image.open(path)).unsqueeze(0)

def get_segmap(index_list, pipe):
    index_tensor_list = [i for i in map(pipe, index_list)]
    segmap = torch.cat(index_tensor_list)
    segmap_ = torch.sum(segmap, 0, keepdim=True)
    return segmap_.unsqueeze(0)
index = 'granularity2'
test_dataset_root_path = '/home/waihinchan/Desktop/scar/dataset/pair/' + index
seg_map_path = '/home/waihinchan/Desktop/scar/dataset/pair/' + index + '/segmap'
test_datasets_folder_path = [os.path.join(test_dataset_root_path,folder) for folder in os.listdir(test_dataset_root_path) if 'pair' in folder]
test_datasets_folder_path.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1].split('pair')[-1]).group(1)))
for img in os.listdir(seg_map_path):
    if 'segmap' in img :
        segmap = get_tensor(os.path.join(seg_map_path,img))
test_datasets_img_path = []
for j,subfolder in enumerate(test_datasets_folder_path,start=0):
    data_path = [os.path.join(subfolder,img) for img in os.listdir(subfolder) if 'jpg' in img and not 'label' in img and not 'single' in img]
    data_path.sort()
    segmap_ = [Image.open(os.path.join(subfolder,img)) for img in os.listdir(subfolder) if 'jpg' in img and 'single' in img]
    if len(segmap_)==0:
        seg = segmap
    else:
        seg = get_segmap(segmap_,pipe)
    rawdata = [get_tensor(path) for path in data_path]
    data = {
        'current': rawdata[0],
        'next': rawdata[-1],
        'last': rawdata[-2],
        'difference': rawdata[1],
        'label': seg
    }
    if fwmode == 'pair':
        fake,label = mymodel.inference(data,segmap_dir=seg_map_path,mode=fwmode)
        imsave(fake, index=str(j) + 'fake', dir='./result/compare/')
        # for i in range(myoption.label_CH):
        #     imsave(label[:,i,:,:], index=str(j) + 'label'+str(i), dir='./result/compare/')
        imsave(data['next'], index=str(j) + 'real', dir='./result/compare/')
        imsave(data['current'], index=str(j) + 'current', dir='./result/compare/')
    elif fwmode=='seq':
        fake_list = mymodel.inference(data,segmap_dir=seg_map_path,mode=fwmode)
        for k,_ in enumerate(fake_list,start=0):
            imsave(_, index=str(k) + 'fake', dir='./result/compare/')
        break