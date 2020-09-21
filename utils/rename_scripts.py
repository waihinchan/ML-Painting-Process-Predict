import os


def rename(path):
    a = os.listdir(path)
    datapath = path
    for i, name in enumerate(a):
        os.rename(os.path.join(datapath,name), str(i)+'.jpg')
        # print(name)

rename('/home/waihinchan/Desktop/scar/dataset/step/_0')