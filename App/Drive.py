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
from export import *
from joint_test import *
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

# class PredThread(Thread):
#     """预测线程"""
#     def __init__(self, model_seg, model_class):
#         super(PredThread, self).__init__()
#         self.DoRun = True
#         self.SegModel = model_seg
#         self.ClassModel = model_class
#         # 用''代替'N'表示None
#         self.press = ''
#
#     def run(self):
#         """Run"""
#         while self.DoRun:
#             img = pyautogui.screenshot(region=[0, 0, 640, 480])  # x,y,w,h
#             # Seg
#             img = alter_predict(model, img, DEVICE)
#             # 归一化
#             img = img / 2
#             img = torch.unsqueeze(img, 0)
#             output = self.ClassModel(img)
#             pred = torch.argmax(output, 1)
#             press = Dict[pred.item()]
#             self.press = press
#
#     def GetPress(self):
#         """
#         取出预测的按键值
#         :return:
#         """
#         return self.press
#
#     def Stop(self):
#         self.DoRun = False


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

    ClassModel = torch.load("./save/drive.pth", map_location=torch.device(DEVICE))

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

    KThread.start()
    # press E to stop
    current = ''
    while True:
        st = time.time()
        img = pyautogui.screenshot(region=[0, 0, 640, 480])  # x,y,w,h
        # Seg
        img = alter_predict(SegModel, img, DEVICE)
        # 归一化
        img = img / 2
        # class
        pred = tensor_predict(ClassModel, img, DEVICE)
        press = Dict[pred.item()]

        if press != current:
            # 更改当前按键
            KThread.ChangeKey(press)
            current = press
        if is_pressed('e'):
            print('end')
            KThread.Stop()
            break

        print("time", time.time() - st)
