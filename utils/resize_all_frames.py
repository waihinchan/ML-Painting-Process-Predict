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

def get_all_floder(root_path):
    if os.path.isdir(root_path):
        floders = [floder for floder in os.listdir(root_path) if not floder.endswith('.DS_Store')]
        # floders = os.listdir(root_path)

        return floders
    else:

        print("%s_is not a valid path!" %root_path)

def get_all_images(root_path):
    if os.path.isdir(root_path):
        images = []
        all = os.walk(root_path)
        for root, dirs, files in all:
            images += [os.path.join(root,name) for name in files if is_image_file(name)]

        return images
    else:
        print("%s_is not a valid path!" %root_path)


def resize_to_suqare(path,desired_size=1024):

    orignal_paths = path[0]
    # list
    save_dir = path[1]
    desired_size = desired_size
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        print("create a new dir %s" % save_dir)

    for orignal_path in orignal_paths:
        im_pth = orignal_path
        im = cv2.imread(im_pth)
        old_size = im.shape[:2] # old_size is in (height, width) format
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        im = cv2.resize(im, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [255, 255, 255]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)

        save_dir_name = os.path.join(save_dir,orignal_path.split('/')[-1])

        # name = video_path.split('/')[-1].split('.')[0]
        cv2.imwrite(save_dir_name,new_im)
        cv2.destroyAllWindows()

def resize_all_frames(path,save_dir = '/home/waihinchan/Desktop/scar/dataset/step_resize'):
    """
    :param path: the video path, can include sub-directory
    :return:
    """
    floders  = get_all_floder(path)

    # this is the all the video folders
    "../dataset/resize_video"
    all_images = [( get_all_images(os.path.join(path,floder)) , os.path.join(save_dir,floder) ) for floder in floders]
    # [([str,str,str],str),([str,str,str],str),([str,str,str],str),([str,str,str],str),([str,str,str],str)]


    from multiprocessing import Pool
    with Pool(cpu_num) as p:
        p.map(resize_to_suqare,all_images)



resize_all_frames('/home/waihinchan/Desktop/scar/dataset/step')