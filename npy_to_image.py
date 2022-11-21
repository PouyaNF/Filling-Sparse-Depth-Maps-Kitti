import multiprocessing
import os
import numpy as np
from joblib import Parallel, delayed
import numpngw




def npy_to_image(idx, files, dir_name='filled_depth/'):

    print(f'image{idx + 1}/ {len(files)} ')

    ##### load numpy array
    item_files = files[idx]
    # rgb_path = os.path.join(kitti_root_path, item_files['rgb'])
    depth_path = os.path.join(kitti_root_path, item_files['depth'])
    depth_path = depth_path.replace('.png', '.npy')
    array = np.load(depth_path)

    ###### preparing array
    array = array.astype(np.uint16)

    ###### directory preparation
    dir = dir_name + item_files['depth']
    img_name = dir.split('/')[-1]
    dir = dir.replace(img_name, '')
    img_name = img_name.replace('.png', '')
    # print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    ### saving the final output as a PNG file
    numpngw.write_png(dir + img_name + '.png', array)




if __name__ == '__main__':

    print("Number of processors: ", multiprocessing.cpu_count())
    num_cores = multiprocessing.cpu_count()
    cam = 2
    mode = 'train'
    files = []  # list of dictionaries
    shared_idx = []
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kitti_root_path = 'D:/Pooya/Dataset/Kitti'

    speed = 8

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

    # print(shared_idx)
    print("all depth maps to be filled: ", len(files))
    print("speed: ", speed)

    # for idx in range(len(files)):
    # save_filled_depth(idx, files, dir_name='filled depth0/')

    Parallel(n_jobs=speed)(delayed(npy_to_image)(i, files) for i in range(len(files)))
