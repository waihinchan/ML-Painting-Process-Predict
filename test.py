from mydataprocess import mydataloader
import option
import net.model
import time
import os
if not os.path.isdir('./result/result_preview'):
  os.mkdir('./result/result_preview')
myoption = option.opt()
myoption.batchSize = 1 # if we have a batch norm maybe this would still working?
myoption.name = 'pair'
myoption.use_degree = 'wrt_position'
myoption.use_label= True
myoption.mode = 'test'
myoption.which_epoch = 400
myoption.forward = 'seq'
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))
mymodel = net.model.SCAR()
mymodel.initialize(opt = myoption)
print(mymodel)
import os
from mydataprocess.dataset import is_image_file
index = '_00010'
path = '/content/scar/dataset/pair/' + index
image_paths = [os.path.join(path,image) for image in os.listdir(path) if is_image_file(image) ]
image_paths.sort()
print(image_paths)
segmap_folder = path + '/segmap'
for _ in os.listdir(segmap_folder):
  if not 'single' in _ and 'segmap' in _:
    label_path = os.path.join(segmap_folder,_)
    print(label_path)
    break
one_hot_paths = [ os.path.join(segmap_folder,one_hot) for one_hot in os.listdir(segmap_folder) if not 'segmap' in one_hot and 'label' in one_hot ]
print(one_hot_paths)
from utils.fast_check_result import grabdata
current = grabdata(image_paths[0],myoption)
last = grabdata(image_paths[-1],myoption)
label = grabdata(label_path,myoption)
data = {
  'current':current,
  'last':last,
  'label':label
}
one_hot_list = [grabdata(one_hot,myoption) for one_hot in one_hot_paths]
fake_frames = mymodel.inference(data,one_hot_list)
print(len(fake_frames))
from utils.fast_check_result import imsave
for i,fake_frame in enumerate(fake_frames,start=1):
  imsave(fake_frame,index =  str(i)+index,dir = './result/result_preview/')
