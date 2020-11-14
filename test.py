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
myoption.shuffle = False
myoption.mode = 'test'
myoption.which_epoch = 100
myoption.forward = 'pair'
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))
mymodel = net.model.SCAR()
mymodel.initialize(opt = myoption)
print(mymodel)
# # *******************single test******************* 
import os
from mydataprocess.dataset import is_image_file
index = '_00000'
path = '/content/scar/dataset/pair/' + index
image_paths = [os.path.join(path,image) for image in os.listdir(path) if is_image_file(image) ]
image_paths.sort()

segmap_folder = path + '/segmap'
for _ in os.listdir(segmap_folder):
  if not 'single' in _ and 'segmap' in _:
    label_path = os.path.join(segmap_folder,_)
    break
one_hot_paths = [ os.path.join(segmap_folder,one_hot) for one_hot in os.listdir(segmap_folder) if not 'segmap' in one_hot and 'label' in one_hot ]

from utils.fast_check_result import grabdata
num = 10
current = grabdata(image_paths[num],myoption)
next_f = grabdata(image_paths[num+1],myoption)
last = grabdata(image_paths[-1],myoption)
label = grabdata(label_path,myoption)
# print(image_paths[0])
# print(image_paths[-1])
# print(label_path)
for _ in one_hot_paths:
  print(_)
data = {
  'current':current,
  'last':last,
  'label':label
}
from utils.fast_check_result import imsave
imsave(current,index = 'current',dir = './result/result_preview/')

one_hot_list = [grabdata(one_hot,myoption) for one_hot in one_hot_paths] if myoption.use_degree is not None else None
for i in range(10):
  fake_frames = mymodel.inference(data,one_hot_list,times = 1)
  imsave(fake_frames[0],index = 'fake'+str(i),dir = './result/result_preview/')
  # imsave(current,index = 'current',dir = './result/result_preview/')
  # imsave(next_f,index = 'next',dir = './result/result_preview/')

# for j,current in enumerate(image_paths,start=1):
#   if j%9==0 or j == 1:
#     current = grabdata(current,myoption)
#     data['current'] = current
#     fake_frames = mymodel.inference(data,one_hot_list,times = 1)
#     imsave(fake_frames[0],index = str(j),dir = './result/result_preview/')


# fake_frames = mymodel.inference(data,one_hot_list,times = 100)
# from utils.fast_check_result import imsave
# for i,fake_frame in enumerate(fake_frames,start=1):
#   imsave(fake_frame,index =  str(i)+index,dir = './result/result_preview/')

# # *******************single test******************* 

# dataloader = mydataloader.Dataloader(myoption)
# pair_data = dataloader.load_data()
# from utils.fast_check_result import imsave
# for j, pair in enumerate(pair_data,start=1):
#       # imsave(pair['current'],index = 'current',dir = './result/result_preview/')
#       fake = mymodel.test(pair)

#       imsave(fake,index = 'fake'+str(j),dir = './result/result_preview/')
#       # if j == 1:
#       #   fake = mymodel.test(pair)
#       #   imsave(fake,index = 'fake'+str(j),dir = './result/result_preview/')
#       # else:
#       #   pair['current'] = fake
#       #   fake = mymodel.test(pair)
#       #   imsave(fake,index = 'fake'+str(j),dir = './result/result_preview/')





