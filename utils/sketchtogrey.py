import cv2
import os
import multiprocessing
cpu_num = multiprocessing.cpu_count()

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_all_sketch_floder(root_path):
    if os.path.isdir(root_path):
        # floders = [os.path.join(root_path,floder) for floder in os.listdir(root_path) if not floder.endswith('.DS_Store')]
        # # floders = os.listdir(root_path)
        # sketch_floders = []
        sketch = []
        all = os.walk(root_path)
        for root, dirs, files in all:
            if root.endswith('sketch'):
                sketch += [root]
        return sketch

    else:
        print("%s_is not a valid path!" %root_path)

allsketch = get_all_sketch_floder('../dataset/video')
all_image = []
for sketch_floder in allsketch:
    for image in os.listdir(sketch_floder):
        image_path = os.path.join(sketch_floder,image)
        if not image_path.endswith('.DS_Store')  :
            all_image.append(image_path )



def greyimage(imagepath):
    image = cv2.imread(imagepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(imagepath,gray)
    # cv2.destroyAllWindows()

from multiprocessing import Pool
with Pool(cpu_num) as p:
    p.map(greyimage,all_image)