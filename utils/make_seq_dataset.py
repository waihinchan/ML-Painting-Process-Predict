# import os
# import re
# import shutil

# #思路：由于pair_datasets是 1to2 1to3 1to4 1to5 2to3 2to4 2to5 2to6
# # 我们需要的是1to2 2to3 + 1to3 2to4 + 1to4 2to5
# # 先排序 然后隔3个取等

# granularity = 3
# def get_folders(path):# get all the video folder
#     return [os.path.join(path,folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path,folder))]
# def get_pairs(path):# get all the pair in ONE video folder
#     pairs = [ os.path.join(path,pair) for pair in os.listdir(path) if 'pair' in pair and os.path.isdir(os.path.join(path,pair))]
#     pairs.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1].split('pair')[-1].split('to')[0]).group(1)))
#     # TODO update this disaster in the future
#     return pairs
# def sort_get_pairs(pair_list,granularity):
#     group_list = []
#     for _ in range(0,len(pair_list),granularity):
#         group = pair_list[_:_+granularity]
#         group.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1].split('pair')[-1].split('to')[-1]).group(1)))
#         group_list.append(group)
#     return group_list

# def make_seq_dataset(list,granularity,target_path):
#     # list format = [[group1],[group2],[group3],[group4]]
#     # group = [itoj,itoj+1,itoj+2,...itoj+granularity]
#     # target_path should be a empty folder like ../dataset/seq
#     video_folder = os.path.join(target_path,target_path,list[0][0].split('/')[-2])
#     if not os.path.isdir(video_folder):
#         os.mkdir(video_folder)
#     granularity_list = []
#     for i in range(granularity):
#         folder = os.path.join(target_path,list[0][0].split('/')[-2]+'/granularity'+str(i))
#         # pick whatever a list and get the video name,
#         if not os.path.isdir(folder):
#             os.mkdir(folder)
#             granularity_list.append(folder)
#     for j,group in enumerate(list,start=0):
#         # j control the freq
#         # this is the No.X group
#         for k,image in enumerate(group,start=0):
#             # k is the the index but also the granularity
#             # this is the No.Z image in No.X group
#             if k==0: # the 0-1 1-2 2-3 group
#                 shutil.copytree(image, os.path.join(granularity_list[k], image.split('/')[-1]))
#             else:
#                 if int(j%(k+1))==0:
#                     shutil.copytree(image, os.path.join(granularity_list[k],image.split('/')[-1]))
#     # for group in list:
#     #     assert len(group) == granularity, 'please confirm the length of your granularity'
#     #     for image,granularity_folder in zip(group,granularity_list):#turple([gran*3],[gran*3])
#     #         shutil.copytree(image, os.path.join(granularity_folder,image.split('/')[-1]))

# def make_all_seq_dataset(source_path,target_path,granularity):
#     all_video_folders = get_folders(source_path)
#     all_video_folder_pairs = [get_pairs(video_folder) for video_folder in all_video_folders]
#     all_video_folder_sort_pairs = [sort_get_pairs(video_folder_pair,granularity) for video_folder_pair in all_video_folder_pairs]
#     for _ in all_video_folder_sort_pairs:
#         for __ in _:
#             print(__)
#     for _ in all_video_folder_sort_pairs:
#         make_seq_dataset(_,granularity,target_path)


# make_all_seq_dataset(source_path='/home/waihinchan/Desktop/scar/dataset/pair',
#                      target_path='/home/waihinchan/Desktop/scar/dataset/seq',
#                      granularity=granularity)

import os
import re
import shutil
granularity = 3
def get_folders(path):# get all the video folder
    video_folders = [os.path.join(path,folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path,folder))]
    all_granularity = []
    for video_folder in video_folders:
      all_granularity+=[os.path.join(video_folder,granularity) for granularity in os.listdir(video_folder) if 'granularity' in granularity]
      # granularity1+2+3+N
    return all_granularity
def get_pairs(path):# get all the pair in ONE video folder
    pairs = [ os.path.join(path,pair) for pair in os.listdir(path) if 'pair' in pair and os.path.isdir(os.path.join(path,pair))]
    pairs.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1].split('pair')[-1].split('to')[0]).group(1)))
    # TODO update this disaster in the future
    return pairs
def sort_get_pairs(pair_list,granularity):
    group_list = []
    for _ in range(0,len(pair_list),granularity):
        group = pair_list[_:_+granularity]
        group.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1].split('pair')[-1].split('to')[-1]).group(1)))
        # get the folder name, get the index before or after the 'to'
        group_list.append(group)
    return group_list
def make_seq_dataset(dataset_root,max_frame = 25):
  paths = get_folders(dataset_root) # this is all the granularity paths
  for data in paths: 
    pairs = get_pairs(data) # get all the pair inside
    print( 'current pairs length')
    print(len(pairs))
    if len(pairs)>max_frame:
      pairs.sort(key=lambda x: int(re.match('(\d+)', x.split('/')[-1].split('pair')[-1].split('to')[0]).group(1)))
      # get the pair folder name, get the index before or after the 'to' then sort it
      cut = max_frame
      # print(cut)
      for i in range(cut,len(pairs),cut): 
        cut_list = pairs[i:i+cut] if i+ cut<=len(pairs) else pairs[i:-1]
        parent = os.path.dirname(cut_list[0]) #  root/video_index/granularityN
        dest = parent+'_'+str((i))
        if not os.path.isdir(dest):
          print(dest)
          os.mkdir(dest) # root/video_index/granularityN_{1,2,3,4..}
        for _ in cut_list:
          # break
          print(_)
          shutil.move(_, dest)

          #move all the pair to dest, so will like root/video_index/granularityN_{1,2,3,4..}/pair{NtoN}

make_seq_dataset('../dataset/pair')







