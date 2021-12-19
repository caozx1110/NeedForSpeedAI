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

class_encoding = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('road', (255, 0, 0)),
    ('car', (0, 255, 0)),
])

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
        'S': 0x1f,
        'D': 0x20
    }

    def __init__(self):
        super(KeyThread, self).__init__()
        self.ScanCode = []
        self.DoRun = True
        self.DoPause = False

    def run(self) -> None:
        """run"""
        while self.DoRun:
            while self.DoPause:
                time.sleep(0.1)

            if self.ScanCode:
                for c in self.ScanCode:
                    if c == self.CodeDic['W']:
                        if random.random() > 0.2:
                            k.key_press(c, 0.005)
                    else:
                        k.key_press(c, 0.005)
                    # k.key_down(c)
                    # # print("press", c)
                # time.sleep(0.001)
                # time.sleep(0.001)

    def ChangeKey(self, key):
        """
        :param key: the list of the pressing key, e.g. 'WA'
        :return: None
        """
        # for _k in self.ScanCode:
        #     k.key_up(_k)
        self.ScanCode = []
        for _k in key:
            self.ScanCode.append(self.CodeDic[_k])

    def Stop(self):
        self.DoRun = False

    def Pause(self):
        self.DoPause = True

    def Restart(self):
        self.DoPause = False


if __name__ == "__main__":
    print("loading model...")
    # 分割网络
    SegModel = ENet(num_classes=3).to(DEVICE)
    checkpoint = torch.load('./save/aug2/nfs_enet', map_location=torch.device(DEVICE))
    SegModel.load_state_dict(checkpoint['state_dict'])
    # 分类网络
    ClassModel = torch.load("./save/max.pth", map_location=torch.device(DEVICE))
    print('model loaded!')
    print(DEVICE)
    # 按键线程
    KThread = KeyThread()

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
        # img.save('./temp.png')
        # Seg
        # img = Image.open('./temp.png')
        img = alter_predict(SegModel, img, DEVICE)
        # render = pre2render(img, class_encoding)
        # plt.imshow(render)
        # plt.show()
        # 归一化
        img = img / 2
        # class
        pred = tensor_predict(ClassModel, img, DEVICE)
        press = Dict[pred.item()]

        print(press)

        if press != current:
            # 更改当前按键
            KThread.ChangeKey(press)
            current = press
        # 按e结束
        if is_pressed('e'):
            print('end')
            KThread.Stop()
            break
        # pause
        if is_pressed('p'):
            print('pause')
            KThread.Pause()
            while True:
                time.sleep(0.1)
                # restart
                if is_pressed('r'):
                    KThread.Restart()
                    break

        # 一个循环用时
        print("time", time.time() - st)


# import cv2
# import matplotlib.pyplot as plt
# SegModel = ENet(num_classes=3).to(DEVICE)
# checkpoint = torch.load('./save/nfs_enet', map_location=torch.device(DEVICE))
# SegModel.load_state_dict(checkpoint['state_dict'])
# # 分类网络
#
# color_encoding = OrderedDict([
#     ('unlabeled', (0, 0, 0)),
#     ('road', (128, 64, 128)),
#     ('car', (64, 0, 128)),
# ])
# ClassModel = torch.load("./save/drive.pth", map_location=torch.device(DEVICE))
# root = '..\\data\\Collect\\data_raw\\czx1'
# img_list = os.listdir(root)
# i = 0
# i = i + 1
# plt.close()
# item = img_list[i]
# img = Image.open(os.path.join(root, item))
# lbl = alter_predict(SegModel, img, DEVICE)
# x = pre2render(lbl, color_encoding)
# plt.subplot(121)
# plt.imshow(img)
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(x)
# plt.axis('off')
# pred = Dict[tensor_predict(ClassModel, lbl / 2, DEVICE).item()]
# print(pred)

