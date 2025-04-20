#-*- coding:utf-8 -*-
import os
import argparse
from data_split_rebuild import data_set_split
from train import main

data_set_split(
        input_folders=['./raw/LOLv1/our485/low'],
        label_folders=['./raw/LOLv1/our485/high'],
        target_folder='./data/LOLv1',
        labeled_scale=0.9
    ) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N')
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--train_batchsize', default=6, type=int, help='train batchsize')
    parser.add_argument('--val_batchsize', default=3, type=int, help='val batchsize')
    parser.add_argument('--crop_size', default=256, type=int, help='crop size')
    parser.add_argument('--resume', default='False', type=str, help='if resume')
    parser.add_argument('--resume_path', default='/path/to/your/net.pth', type=str, help='if resume')
    parser.add_argument('--use_pretain', default='False', type=str, help='use pretained model')
    parser.add_argument('--pretrained_path', default='/path/to/pretained/net.pth', type=str, help='if pretrained')
    parser.add_argument('--data_dir', default='./data/LOLv1', type=str, help='data root path')
    parser.add_argument('--save_path', default='./model/ckpt/', type=str)
    parser.add_argument('--log_dir', default='./model/log', type=str)

    args = parser.parse_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    main(-1, args)
