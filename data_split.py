# -*- coding: utf-8 -*-

import os
import random
from shutil import copy2

randnum = 2025

def data_set_split(src_folder, target_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    '''
    :src_folder: 源文件夹(源文件夹下只能有一层文件夹,如：src_folder/high/1.png)
    :target_folder: 目标文件夹
    :train_scale: 训练集比例
    :val_scale: 验证集比例
    :test_scale: 测试集比例
    '''
    print('start dataset splitting')
    class_names = os.listdir(src_folder)
    split_names = ['train', 'val', 'test']

    for split_name in split_names:
        split_path = os.path.join(target_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)

        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    for class_name in class_names:
        current_class_path = os.path.join(src_folder, class_name)
        if not os.path.isdir(current_class_path):
            continue
        current_data_folder = os.listdir(current_class_path)
        current_data_length = len(current_data_folder)
        current_data_index_list = list(range(current_data_length))
        random.seed(randnum)
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_folder, 'val'), class_name)
        test_folder = os.path.join(os.path.join(target_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num, val_num, test_num = 0, 0, 0

        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_path, current_data_folder[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                train_num = train_num + 1
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                copy2(src_img_path, val_folder)
                val_num = val_num + 1
            else:
                copy2(src_img_path, test_folder)
                test_num = test_num + 1
            current_idx = current_idx + 1

        print("********************************")
        print("train set{}: {}".format(train_folder, train_num))
        print("val set{}: {}".format(val_folder, val_num))
        print("test set{}: {}".format(test_folder, test_num))

src_folder = './data/raw/LOL/our485' 
target_folder = './data/data_splited'
data_set_split(src_folder, target_folder)
        