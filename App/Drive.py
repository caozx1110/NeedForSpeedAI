# coding=utf-8
"""
@author: Cao Zhanxiang
@project: PreTest
@file: Drive.py
@date: 2021/12/7
@function: 使用模型预测操作，模拟按键输入
"""

import torch
import pyautogui
import time
import random
from DriveNet import DriveNet
from ENet import ENet
from PIL import Image
from threading import Thread
# 两个编好的预测函数
from export import alter_predict, tensor_predict
from keyboard import is_pressed
import KeyboardEmulation as k

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# label to key dict
Dict = {'A': 0, 'D': 1, 'N': 2, 'W': 3, 'WA': 4, 'WD': 5, 0: 'A', 1: 'D', 2: 'N', 3: 'W', 4: 'WA', 5: 'WD'}

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
                    # 如果预测值为W，则以一定概率不按W
                    if c == self.CodeDic['W']:
                        if random.random() > 0.2:
                            k.key_press(c, 0.005)
                    else:
                        k.key_press(c, 0.005)

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
