from utils import dataset_test
from utils import fast_check_result
import torch
from net.generator import Decoder2
import option
# my = option.opt()
# for name,value in vars(my).items():
#     print('%s=%s' % (name,value))
# a = Decoder2(my)
# z = torch.rand(1,1,256)
# input = torch.rand(1,8,my.input_size,my.input_size)
# print(a(input,z))

def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1)



test_tensor = fast_check_result.grabdata('/home/waihinchan/Desktop/scar/dataset/pair/_00001/segmap/singlesegmap_0003_eyebrow.jpg')
# test_tensor[test_tensor>=0.5]=1
# test_tensor[test_tensor<0.5]=0
print(test_tensor.unique())
test_tensor = fast_check_result.grabdata('/home/waihinchan/Desktop/SPADE/datasets/coco_stuff/train_inst/000000017914.png')
print(test_tensor.unique(return_counts=True))





