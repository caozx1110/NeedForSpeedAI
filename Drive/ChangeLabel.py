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




