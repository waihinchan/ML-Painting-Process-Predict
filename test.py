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
myoption.which_epoch = 200
myoption.forward = 'pair'
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))
mymodel = net.model.SCAR()
mymodel.initialize(opt = myoption)
print(mymodel)
# # *******************single test******************* 
import os
from mydataprocess.dataset import is_image_file
index = '_00001'
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
num = 0
current = grabdata(image_paths[num],myoption)
next_f = grabdata(image_paths[num+1],myoption)
last = grabdata(image_paths[-1],myoption)
label = grabdata(label_path,myoption)
# print(image_paths[0])
# print(image_paths[-1])
# print(label_path)
for _ in one_hot_paths:
  print(_)

from utils.fast_check_result import imsave
one_hot_list = [grabdata(one_hot,myoption) for one_hot in one_hot_paths] if myoption.use_degree is not None else None
data = {
  'current':current,
  'next':current,
  'last':last,
  'label':label,
  'difference':current,
  'segmaps':one_hot_list
}

 # save 10 groups of different random degree result
fake_frames = mymodel.inference(data,100)
if not os.path.isdir('./result/result_preview/' + 'frames'+str(20)):
  os.mkdir('./result/result_preview/' + 'frames'+str(20))
for j,fake_frame in enumerate(fake_frames,start=0):
  imsave(fake_frame,index = str(j),dir = './result/result_preview/' + 'frames'+str(20) + '/')
  imsave(current,index = 'current',dir = './result/result_preview/'+ 'frames'+str(20) + '/')

# # *******************single test******************* 

# dataloader = mydataloader.Dataloader(myoption)
# pair_data = dataloader.load_data()
# from utils.fast_check_result import imsave
# for j, pair in enumerate(pair_data,start=1):
#       # imsave(pair['current'],index = 'current',dir = './result/result_preview/')

#       # destory the structure and make a test
#       pair['difference'] = pair['current']
#       pair['next'] = pair['current']
#       # destory the structure and make a test
#       fake = mymodel.test(pair)
#       imsave(fake,index = str(j)+'fake',dir = './result/result_preview/')
#       imsave(pair['difference'],index = str(j)+'difference',dir = './result/result_preview/')
#       imsave(pair['next'],index = str(j)+'next',dir = './result/result_preview/')
#       imsave(pair['current'],index = str(j)+'current',dir = './result/result_preview/')
#       imsave(pair['label'],index = str(j)+'label',dir = './result/result_preview/')

#       # if j == 1:
#       #   fake = mymodel.test(pair)
#       #   imsave(fake,index = 'fake'+str(j),dir = './result/result_preview/')
#       # else:
#       #   pair['current'] = fake
#       #   fake = mymodel.test(pair)
#       #   imsave(fake,index = 'fake'+str(j),dir = './result/result_preview/')





