import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def rename(path):
    a = os.listdir(path)
    datapath = path

    # for i, name in enumerate(a):
    #     os.rename(os.path.join(datapath,name), str(i)+'.png')
        # print(name)
def get_all_images(root_path):
    if os.path.isdir(root_path):
        sub_dirs = []
        sub_dirs = [os.path.join(root_path,sub_dir) for sub_dir in os.listdir(root_path) if sub_dir != '.DS_Store']
        for sub_dir in sub_dirs:
            images = sorted([os.path.join(sub_dir,image) for image in os.listdir(sub_dir) if is_image_file(image)])
            for i, name in enumerate(images):
                # print(name.split('/')[:-1])
                os.rename(name, os.path.join(os.path.dirname(name),str(i)+'.jpg'))

    else:
        print("%s_is not a valid path!" %root_path)

get_all_images('/Users/waihinchan/Documents/mymodel/scar/dataset/resize_step')
