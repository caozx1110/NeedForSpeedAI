# coding=utf-8
"""
@author: Cao Zhanxiang
@project: PreTest
@file: Drive.py
@date: 2021/12/7
@function: 使用模型
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import pyautogui
import time
from DriveNet import DriveNet
from enet import ENet
from torch import nn
from PIL import Image
from torchvision import transforms as F
from threading import Thread
from export import alter_predict
from keyboard import *
import KeyboardEmulation as k


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# label to key dict
# Dict = {
#     'W': 0,
#     'A': 1,
#     'S': 2,
#     'D': 3,
#     'WA': 4,
#     'WD': 5,
#     'AS': 6,
#     'SD': 7,
#     'N': 8,
#     0: 'W',
#     1: 'A',
#     2: 'S',
#     3: 'D',
#     4: 'WA',
#     5: 'WD',
#     6: 'AS',
#     7: 'SD',
#     8: ''
# }

Dict = {'A': 0, 'D': 1, 'N': 2, 'W': 3, 'WA': 4, 'WD': 5, 0: 'A', 1: 'D', 2: 'N', 3: 'W', 4: 'WA', 5: 'WD'}

class PredThread(Thread):
    """预测线程"""
    def __init__(self, model_seg, model_drive):
        super(PredThread, self).__init__()
        self.DoRun = True
        self.SegModel = model_seg
        self.DriveModel = model_drive
        # 用''代替'N'表示None
        self.press = ''

    def run(self):
        """Run"""
        while self.DoRun:
            img = pyautogui.screenshot(region=[0, 0, 640, 480])  # x,y,w,h
            # NO!
            # 640 x 480 -> 640 x 640
            # img = process_each_pic(img)
            # Seg
            st = time.time()
            img = alter_predict(self.SegModel, img, DEVICE)
            print('seg', time.time() - st)
            st = time.time()
            # 归一化
            img = img / 2
            # img = F.ToTensor()(img)
            # 1 x 640 x 640
            img = torch.unsqueeze(img, 0)
            output = self.DriveModel(img)
            pred = torch.argmax(output, 1)
            press = Dict[pred.item()]
            print('drive', time.time() - st)
            self.press = press

    def GetPress(self):
        """
        取出预测的按键值
        :return:
        """
        return self.press

    def Stop(self):
        self.DoRun = False

# pad to 640 x 640
def process_each_pic(img):
    img = img.convert('RGB')
    w, h = img.size
    output_img = Image.new('RGB', size=(max(w, h), max(w, h)), color=(0, 0, 0))
    length = int(abs(w - h) // 2)
    box = (length, 0) if w < h else (0, length)
    output_img.paste(img, box)
    return output_img

class KeyThread(Thread):
    """按键线程"""
    CodeDic = {
        'W': 0x11,
        'A': 0x1e,
        'F': 0x1f,
        'D': 0x20
    }

    def __init__(self):
        super(KeyThread, self).__init__()
        self.ScanCode = []
        self.DoRun = True

    def run(self) -> None:
        """run"""
        while self.DoRun:
            if self.ScanCode:
                for c in self.ScanCode:
                    k.key_down(c)

    def ChangeKey(self, key):
        """
        :param key: the list of the pressing key, e.g. 'WA'
        :return: None
        """
        self.ScanCode = []
        for _k in key:
            self.ScanCode.append(self.CodeDic[_k])

    def Stop(self):
        self.DoRun = False


if __name__ == "__main__":
    SegModel = ENet(num_classes=3).to(DEVICE)
    checkpoint = torch.load('./save/nfs_enet', map_location=torch.device(DEVICE))
    SegModel.load_state_dict(checkpoint['state_dict'])

    DriveModel = torch.load("./save/drive.pth", map_location=torch.device(DEVICE))
    # screen shoot and predict
    PThread = PredThread(SegModel, DriveModel)
    KThread = KeyThread()

    # k = PyKeyboard()

    # press b to start
    while True:
        if is_pressed('b'):
            break
    '''example:
    while True:
        # 无敌！！！！！！！！！！！！！！！！！！！！！！！！！！
        k.key_down(0x1e)   # w
        k.key_down(0x11)
        if is_pressed('e'):
            break
    '''

    PThread.start()
    KThread.start()
    # press E to stop
    current = ''
    while True:
        temp = PThread.GetPress()
        # print(temp)
        # temp = 'W'
        if temp != current:
            # 更改当前按键
            KThread.ChangeKey(temp)
            current = temp
        if is_pressed('e'):
            print('end')
            KThread.Stop()
            PThread.Stop()
            break
