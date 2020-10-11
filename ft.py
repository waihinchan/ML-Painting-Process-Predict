# ********************************** this is for test the dataset working **********************************

from mydataprocess import mydataloader
import torch
import option
from utils import fast_check_result
# myoption = option.opt()
# myoption.name = 'pair'
# myoption.input_size = 256
# for name,value in vars(myoption).items():
#     print('%s=%s' % (name,value))
# dataloader = mydataloader.Dataloader(myoption)
# thedataset = dataloader.load_data()
# for i, data in enumerate(thedataset,start=0):
#     # if i%500==0:
#     current = data['current']
#     last = data['last']
#     difference = data['difference']
#     next = data['next']
#     label = data['label']
#
#     # fast_check_result.imsave(data, dir='./result/data_result/', index=str(i))
#     fast_check_result.imsave(next, dir='./result/data_result/', index=str(i) + '-' + 'next')
#     fast_check_result.imsave(current, dir='./result/data_result/', index=str(i) + '-' + 'current')
#     fast_check_result.imsave(last, dir='./result/data_result/', index=str(i) + '-' + 'last')
#     fast_check_result.imsave(difference, dir='./result/data_result/', index=str(i) + '-' + 'difference')
#     fast_check_result.imsave(label, dir='./result/data_result/', index=str(i) + '-' + 'label')


a = torch.rand(1,6,512,512)
print(a.size(0))
# from utils.fast_check_result import grabdata
# from utils.fast_check_result import imsave
# import torch
# step = 255
# label_mask = grabdata(myoption,'/home/waihinchan/Desktop/scar/dataset/pair/_0/_0pair8/difference8.jpg')
# pool = torch.nn.MaxPool2d(50, stride=2, padding=[1, 1])
# result = pool(label_mask)
# imsave(result, dir='./result/data_result/', index='ds')
#
# print(result.shape)
# for i in range(3):
#     result = pool(result)
#     print(result.shape)
#     imsave(result, dir='./result/data_result/', index='ds'+ str(i))

# print(label_mask)
# a = 0.3827 * 100
# print(a)
# print(round(a))
# print(round(a/10))