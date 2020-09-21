# this is for test the generator and encoder
# import torch
# import net.generator
# import option
# myoption  = option.opt()
# test_generator = net.generator.global_frame_generator(opt=myoption,input_channel=6)
# random_input = torch.rand(1,6,512,512)
# print("test_generator")
# print(test_generator)
# print("-----------------------outputshape---------------------")
# print(test_generator(random_input).shape)

# test_encoder = net.generator.Encoder(opt = myoption,input_channel = myoption.input_chan*2)
# print("test_Encoder")
# print(test_encoder)
# print("-----------------------outputshape---------------------")
# if myoption.use_vector:
#     print("mu")
#     print(test_encoder(random_input)[0].shape)
#     print("var")
#     print(test_encoder(random_input)[1].shape)
# else:
#     print(test_encoder(random_input).shape)

# this is for test the generator and encoder

# ********************************** this is for test the dataset working **********************************

from mydataprocess import mydataloader
import torch
import option
from utils import fast_check_result
myoption = option.opt()
myoption.name = 'step'
myoption.bs_total_frames = 5
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

dataloader = mydataloader.Dataloader(myoption)

thedataset = dataloader.load_data()
once = True
i=0

for data in thedataset:
    i+=1
    if once:
        j=0
        frames = data['frames']
        for frame in frames:
            fast_check_result.imsave(frame, dir='./result/data_result2/', index=str(i)+'-'+str(j))
            j+=1
        last_frames = data['target']
#         print('******************** printing frames shape ********************')
#         print(frames[-1].shape)
#         print('******************** printing last frame shape ********************')
#         print(last_frames.shape)
        fast_check_result.imsave(last_frames,dir = './result/data_result2/',index=str(i)+('target'))
        # once = False

# ********************************** this is for test the dataset working **********************************

# A = torch.rand(1,3,150,150)
# B = torch.rand(1,3,150,150)
# C = torch.rand(1,3,150,150)
# print(torch.cat([A,B,C],dim=1).shape)

# this is for test the frames matching
# a = []
# b = []
# c = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# d = []
# last = 5
# for j in range(0,last):
#     a+=['blank image']
#     d+=['blank image']
#
# for i in range(0,20):
#     if i < last:
#         b = a[0:last]
#         print(str(i)+"times b")
#         a[i] = i
#         d[i] = c[i]
#         print(b)
#         print(a)
#         print(d)
#     else:
#         b = a[i-last:i]
#         d = c[i-last:i]
#         a.append(i)
#         print("out of the last")
#         print(str(i)+"times b")
#         print(b)
#         print(a)
#         print(d)
# this is for test the frames matching

#this is for test the whole model

# import option
# import net.model
# import torch
# myoption = option.opt()
# myscar = net.model.SCAR()
# myscar.initialize(opt = myoption)
# fake_frames_num = 300
# all_frames = []
# for i in range(0,fake_frames_num):
#     all_frames+=[torch.rand(1,3,512,512)]
# target = all_frames[-1]
# input = {
#     'target':target,
#     'frames':all_frames
# }
# print("total frame")
# print(len(input['frames']))
# print("single frame shape")
# print(input['frames'][0].shape)
# print(input['target'].shape)
# print("whole model")
# print(myscar)
# print("single forward test")
# print(myscar(input))

