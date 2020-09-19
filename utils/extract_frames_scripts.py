import cv2
import os
import multiprocessing

extract_frames_num = 300
cpu_num = multiprocessing.cpu_count()

VIDEO_EXTENSIONS = [
    '.mp4', '.AVI', '.MPEG', '.JPEG',
    '.wmv', '.m4v','MP4',
]


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def grab_all_video(path):
    if os.path.isdir(path):
        videos = []
        all = os.walk(path)
        for root, dirs, files in all:
            videos += [os.path.join(root,name) for name in files if is_video_file(name)]

        return videos


    else:
        print("%s_is not a valid path!" %path)


def get_frame(paths):
    video_path = paths[0]
    save_dir = paths[1]
    name = video_path.split('/')[-1].split('.')[0]
    folder = os.path.join(save_dir,name)
    print('create a folder %s' %folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)
        print('create a folder %s' %folder)
    cap = cv2.VideoCapture(video_path)
    i=0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT ))
    frequence = length//extract_frames_num
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i%frequence==0:
            cv2.imwrite(folder+'/'+str(int(i/frequence))+'.jpg',frame)
        i+=1
    cap.release()
    cv2.destroyAllWindows()



def get_all_frames(path,save_dir='../dataset/video'):
    """
    :param path: the videos dirs
    :param save_dir: the dataset root
    :return: None
    """
    videos = grab_all_video(path)
    print(videos)
    videos = [(video,save_dir) for video in videos]
    from multiprocessing import Pool
    with Pool(cpu_num) as p:
        p.map(get_frame,videos)



get_all_frames('/Users/waihinchan/Desktop/mydataset/bilibli_process')


