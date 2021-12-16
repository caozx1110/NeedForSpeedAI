# coding=utf-8
"""
@author: Cao Zhanxiang
@project: Drive
@file: ChangeLabel.py
@date: 2021/12/14
@function: label 1 -> road, label 2 -> car
"""

import os
import json

root = './data_raw/xyx1/data/'

if __name__ == '__main__':
    for file in os.listdir(root):
        path = os.path.join(root, file)
        with open(path, "r", encoding='utf-8') as f:
            data = json.loads(f.read())
            for d in range(len(data['shapes'])):
                if data['shapes'][d]['label'] == '1':
                    data['shapes'][d]['label'] = 'road'
                if data['shapes'][d]['label'] == '2':
                    data['shapes'][d]['label'] = 'car'
            s = json.dumps(data)
            f.close()
        with open(path, "w", encoding='utf-8') as f:
            f.write(s)
            f.close()


import matplotlib.pyplot as plt
from PIL import Image
root = 'dataset\\dataset_mcl'
json_list = os.listdir(root)
i = 0
i += 1
plt.close()
path = os.path.join(root, json_list[i])
print(json_list[i])
img_path = os.path.join(path, 'img.png')
lbl_path = os.path.join(path, 'label_viz.png')
plt.subplot(121)
plt.imshow(Image.open(img_path))
plt.axis('off')
plt.subplot(122)
plt.imshow(Image.open(lbl_path))
plt.axis('off')

