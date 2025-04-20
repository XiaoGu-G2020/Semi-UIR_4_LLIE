# -*- coding: utf-8 -*-

import os
import random
from shutil import copy2
from PIL import Image
from estimate_illumination import luminance_estimation
from glob import glob
import torch
from torchvision.transforms import transforms

random.seed(114514)

split_names = ['labeled', 'unlabeled', 'val', 'test']
paired_classes = ['input', 'GT', 'LA']
unpaired_classes = ['input', 'candidate', 'LA']
def _create_folders(root_folder: str):
    for split_name in split_names:
        split_path = os.path.join(root_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.makedirs(split_path, exist_ok = True)

        if split_name == 'unlabeled':
            class_names = unpaired_classes
        else:
            class_names = paired_classes

        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.makedirs(class_split_path, exist_ok = True)

def _get_LAimage(input_image_path: str, LAcache: str = './LAcache'):
    # 输入源文件路径，检查LA缓存中是否有同名文件，有的话返回一个LA文件路径，没有的话就生成一个LA，然后返回LA文件路径
    if not os.path.exists(os.path.join(LAcache, input_image_path.split('/')[-1])):
        img = Image.open(input_image_path)
        L = luminance_estimation(img)
        ndar = Image.fromarray(L)
        ndar.save(os.path.join(LAcache, input_image_path.split('/')[-1]))
    return os.path.join(LAcache, input_image_path.split('/')[-1])

def _create_candidate(input_image_path: str, target_dir: str):
        img = torch.zeros((3,256,256))
        img_name = input_image_path.split('/')[-1]
        toPil = transforms.ToPILImage()
        res = toPil(img).convert('RGB')
        res.save(os.path.join(target_dir, img_name))

def data_set_split(input_folders, label_folders, target_folder: str, LAcache='./LAcache', train_scale=0.8, val_scale=0.1, test_scale=0.1, labeled_scale=0.2):
    # 先遍历所有input_folder，然后获取input的数量，然后计算出train val test 的数量，然后按照比例进行划分
    LAcache = LAcache
    _create_folders(target_folder)
    input_num = 0
    for input_folder, label_folder in zip(input_folders, label_folders):
        # 判断配对数据集合法
        if len(os.listdir(input_folder)) != len(os.listdir(label_folder)) and len(os.listdir(input_folder)) != 0:
            raise ValueError('{input_folder} and {label_folder} must have the same number of files')
        input_num += len(os.listdir(input_folder))
    
    labeled_stop_flag = int(input_num * train_scale * labeled_scale)
    unlabeled_stop_flag = int(input_num * train_scale)
    val_stop_flag = int(input_num * (train_scale + val_scale))
    current_idx = 0
    labeled_num, unlabeled_num, val_num, test_num = 0, 0, 0, 0

    for input_folder, label_folder in zip(input_folders, label_folders):
        src_input_list = os.listdir(input_folder)
        src_label_list = os.listdir(label_folder)
        # 随机打乱
        random.shuffle(src_input_list)

        for file_name in src_input_list:
            src_input_path = os.path.join(input_folder, file_name)
            src_label_path = os.path.join(label_folder, file_name)
            if current_idx <= labeled_stop_flag:
                # 配对数据集
                copy2(src_input_path, os.path.join(target_folder, 'labeled', 'input'))
                copy2(src_label_path, os.path.join(target_folder, 'labeled', 'GT'))
                LAcache_path = _get_LAimage(src_input_path, LAcache=LAcache)
                copy2(LAcache_path, os.path.join(target_folder, 'labeled', 'LA'))
                labeled_num = labeled_num + 1
            elif (current_idx > labeled_stop_flag) and (current_idx <= unlabeled_stop_flag):
                # 未配对数据集
                copy2(src_input_path, os.path.join(target_folder, 'unlabeled', 'input'))
                _create_candidate(src_input_path, os.path.join(target_folder, 'unlabeled', 'candidate'))
                LAcache_path = _get_LAimage(src_input_path, LAcache=LAcache)
                copy2(LAcache_path, os.path.join(target_folder, 'unlabeled', 'LA'))
                unlabeled_num = unlabeled_num + 1
            elif (current_idx > unlabeled_stop_flag) and (current_idx <= val_stop_flag):
                # 验证集
                copy2(src_input_path, os.path.join(target_folder, 'val', 'input'))
                copy2(src_label_path, os.path.join(target_folder, 'val', 'GT'))
                LAcache_path = _get_LAimage(src_input_path, LAcache=LAcache)
                copy2(LAcache_path, os.path.join(target_folder, 'val', 'LA'))
                val_num = val_num + 1
            else:
                # 测试集
                copy2(src_input_path, os.path.join(target_folder, 'test', 'input'))
                copy2(src_label_path, os.path.join(target_folder, 'test', 'GT'))
                LAcache_path = _get_LAimage(src_input_path)
                copy2(LAcache_path, os.path.join(target_folder, 'test', 'LA'))
                test_num = test_num + 1
            current_idx = current_idx + 1

def main():
    # LAcache 部分测试
    # path = get_LAimage('./raw/LOLv2/Real_captured/Train/Low/00144.png')
    # print(path)

    # data_set_split(
    #     input_folders=['./raw/LOLv2/Real_captured/Train/Low'],
    #     label_folders=['./raw/LOLv2/Real_captured/Train/Normal'],
    #     target_folder='./data/LOLv2',
    #     labeled_scale=0.1
    # ) 
    data_set_split(
        input_folders=['./raw/LOLv1/our485/low'],
        label_folders=['./raw/LOLv1/our485/high'],
        target_folder='./data/LOLv1',
        labeled_scale=0.7
    ) 

if __name__ == '__main__':
    main()