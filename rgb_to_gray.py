import multiprocessing
from Dataloader.filldepth import fill_depth_colorization
from Dataloader.interpd import interpdepth
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.image
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from PIL import Image , ImageOps
from skimage import io
import torchvision.transforms.functional as F


def save_gray_depth(idx, files, dir_name='filled_depth/'):
    item_files = files[idx]
    depth_path = os.path.join(kitti_root_path, item_files['depth'])
    depth = Image.open(depth_path).convert('RGB')
    depth = F.to_grayscale(depth)
    #depth = ImageOps.grayscale(depth)

    print(f'image{idx + 1}/ {len(files)} ')

    dir = dir_name + item_files['depth']
    img_name = dir.split('/')[-1]
    dir = dir.replace(img_name, '')
    # print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    matplotlib.image.imsave(dir + img_name, depth, cmap= 'gray')




if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    print("Number of processors: ", num_cores)

    cam = 2
    mode = 'test'
    files = []  # list of dictionaries
    shared_idx = []
    kitti_root_path = 'D:/Pooya/Dataset/Kitti/All Depths/KITTI_Depth - rgb'
    speed = 6   # max 8


    currpath = os.path.dirname(os.path.realpath(__file__))
    filepath = currpath + '/filenames/eigen_{}_files.txt'.format(mode)
    shared_path = currpath + '/filenames/eigen692_652_shared_index.txt'
    with open(filepath, 'r') as f:
        data_list = f.read().split('\n')  # returns a list , split string at newline
        for data in data_list:
            if len(data) == 0:
                continue
            data_info = data.split(' ')
            assert cam == 2 or cam == 3, "Panic::Param 'cam' should be 2 or 3"
            data_idx_select = (0, 1)[cam == 3]  # if cam = 3 returns 1 else 0
            assert cam == 2 or cam == 3, "Panic::Param 'cam' should be 2 or 3"
            data_idx_select = (0, 1)[cam == 3]  # if cam = 3 returns 1 else 0

            files.append({
                "rgb": data_info[data_idx_select],
                "depth": data_info[data_idx_select + 2]
            })

    # print(files[0]['rgb'])
    # print(files[0]['depth'])

    with open(shared_path, 'r') as f:
        shared_list = f.read().split('\n')
        for item in shared_list:
            if len(item) == 0:
                continue
            shared_idx.append(int(item))

    with open(shared_path, 'r') as f:
        shared_list = f.read().split('\n')
        for item in shared_list:
            if len(item) == 0:
                continue
            shared_idx.append(int(item))

    # print(shared_idx)
    print("all images  to be converted: ", len(files))
    print("speed: ", speed)

    Parallel(n_jobs=speed)(delayed(save_gray_depth)(i, files) for i in range(len(files)))
