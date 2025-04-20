import cv2
import os
import numpy as np
from PIL import Image
from glob import glob
from os.path import join

# estimate the illumination map

def luminance_estimation(img):
    sigma_list = [15, 60, 90]
    img = np.uint8(np.array(img))
    illuminance = np.ones_like(img).astype(np.float32)
    for sigma in sigma_list:
        illuminance1 = np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1e-8)
        illuminance1 = np.clip(illuminance1, 0, 255)
        illuminance = illuminance + illuminance1
    illuminance = illuminance / 3
    L = (illuminance - np.min(illuminance)) / (np.max(illuminance) - np.min(illuminance) + 1e-6)
    L = np.uint8(L * 255)
    return L

def main():
    # input_dir = "data/test/unlabeled/input"
    # 这里一定要指向名为input的文件夹
    input_dir = 'data/test/input'
    input_lists = glob(join(input_dir, "*.*"))
    # result_dir = "data/test/unlabeled/LA/"
    # 把input_dir的路径修改一下，最后的input换成LA，然后进行存在检测，不存在则创建
    result_dir = input_dir.replace('input', 'LA')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for gen_path in zip(input_lists):
        img = Image.open(gen_path[0])
        img_name = gen_path[0].split('/')[-1].split('\\')[-1]
        L = luminance_estimation(img)
        ndar = Image.fromarray(L)
        ndar.save(os.path.join(result_dir, img_name))

    print('finished!')

if __name__ == '__main__':
    main()