from mydataprocess import mydataloader
import option
from utils import fast_check_result
import torch
"""
please don't run this otherwise the root path will error
"""
def test_dataset():
    myoption = option.opt()
    myoption.name = 'pair'
    myoption.use_degree = 'wrt_position'
    myoption.input_size = 256
    myoption.use_label = True
    myoption.forward = 'pair'
    myoption.shuffle = False
    for name,value in vars(myoption).items():
        print('%s=%s' % (name,value))
    dataloader = mydataloader.Dataloader(myoption)
    thedataset = dataloader.load_data()
    save_dir = './result/data_result/'
    for i, data in enumerate(thedataset,start=0):
        # if i%50==0:
        difference = data['difference']
        temp = torch.sum(difference,1,keepdim=True)

        # print(temp.shape)
        for _ in data['segmaps']:
            howmuch = len(_[temp!=0].nonzero())
            itself = len(_.nonzero())

            print('itself')
            print(itself)
            print('**********************')
            print('howmuch')
            print(howmuch)
            print('percentage')
            print(howmuch//itself)

        break


    print('see the result at %s ' % save_dir )