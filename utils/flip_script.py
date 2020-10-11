import os
import multiprocessing
cpu_num = multiprocessing.cpu_count()
from PIL import Image
import PIL



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_all_floder(root_path):
    if os.path.isdir(root_path):
        floders = [floder for floder in os.listdir(root_path) if not floder.endswith('.DS_Store')]
        # floders = os.listdir(root_path)

        return floders
    else:

        print("%s_is not a valid path!" %root_path)

def get_all_flip_images(root_path):
    if os.path.isdir(root_path):
        images = []
        all = os.walk(root_path)
        for root, dirs, files in all:
            # images += [os.path.join(root,name) for name in files if root.endswith('flip') and is_image_file(name)]
            images += [os.path.join(root, name) for name in files if 'flip' in name and is_image_file(name)]
        print(images)
        return images
    else:
        print("%s_is not a valid path!" %root_path)


def flip_images(path):
    input_image = Image.open(path)
    out = input_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    name = path.split('/')[-1]
    top_dir = os.path.dirname(os.path.dirname(path))
    save_dir = os.path.join(top_dir,name)
    out.save(save_dir)


def flip_all_images(root_path):
    all_flip_images = get_all_flip_images(root_path)
    from multiprocessing import Pool
    with Pool(cpu_num) as p:
        p.map(flip_images,all_flip_images)

flip_all_images('../dataset/pair/_0')