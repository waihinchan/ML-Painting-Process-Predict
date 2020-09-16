import os


def rename(path):
    a = os.listdir(path)
    datapath = path
    for i, name in enumerate(a):
        os.rename(os.path.join(datapath,name), str(i)+'.png')
        # print(name)

