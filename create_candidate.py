import os
import torch
from glob import glob
from os.path import join
from torchvision.transforms import transforms

# initialize the reliable bank

# input_dir = 'data/unlabeled/input'
# result_dir = 'data/unlabeled/candidate'
input_dir = 'data/LOLv1/unlabeled/input'
result_dir = 'data/LOLv1/unlabeled/candidate'
if result_dir not in os.listdir(input_dir):
    os.makedirs(result_dir)
input_lists = glob(join(input_dir, '*.*'))
for gen_path in zip(input_lists):
    img = torch.zeros((3,256,256))
    img_name = gen_path[0].split('/')[-1].split('\\')[-1]
    print(img_name)
    toPil = transforms.ToPILImage()
    res = toPil(img).convert('RGB')
    res.save(os.path.join(result_dir, img_name))
