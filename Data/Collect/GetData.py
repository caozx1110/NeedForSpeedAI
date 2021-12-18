# coding=utf-8
"""
@author: Cao Zhanxiang
@project: 8-Queens.py
@file: GetData.py
@date: 2021/11/14
@function: 
"""
import time

# import win32con
# import win32gui
# import win32ui
# from HookThread import HookThread
from keyboard import *
# from PIL import Image
# import threading
import os
import pyautogui
# from cv2 import cv2
# import numpy as np


# https://blog.csdn.net/sinat_38682860/article/details/109388556
# def window_capture(filename):
#     hwnd = 0  # 窗口的编号，0号表示当前活跃窗口
#     # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
#     hwndDC = win32gui.GetWindowDC(hwnd)
#     # 根据窗口的DC获取mfcDC
#     mfcDC = win32ui.CreateDCFromHandle(hwndDC)
#     # mfcDC创建可兼容的DC
#     saveDC = mfcDC.CreateCompatibleDC()
#     # 创建bigmap准备保存图片
#     saveBitMap = win32ui.CreateBitmap()
#     # 获取监控器信息
#     # MoniterDev = win32api.EnumDisplayMonitors(None, None)
#     # w = MoniterDev[0][2][2]
#     # h = MoniterDev[0][2][3]
#     w = 640
#     h = 480
#     # print(MoniterDev)     # 图片大小
#     # 为bitmap开辟空间
#     saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
#     # 高度saveDC，将截图保存到saveBitmap中
#     saveDC.SelectObject(saveBitMap)
#     # 截取从左上角（0，0）长宽为（w，h）的图片
#     saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
#     saveBitMap.SaveBitmapFile(saveDC, filename)


# def ChangeSize(filename):
#     pass
#     img = Image.open(filename)
#     change = img.resize((640, 480))
#     change.save(filename)


if __name__ == '__main__':
    folder = './data_raw/lph2/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Dic = {'W': 0, 'A': 0, 'S': 0, 'D': 0, 'WA': 0, 'WD': 0, 'AS': 0, 'SD': 0, 'N': 0}
    Dic = {'W': 869, 'A': 84, 'S': 0, 'D': 34, 'WA': 120, 'WD': 134, 'AS': 0, 'SD': 0, 'N': 0}
    # bt = time.time()
    # print("begin", bt)
    while True:
        if is_pressed('b'):
            break

    while True:
        Pushed = ''
        if is_pressed('w'):
            Pushed += 'W'
        if is_pressed('a'):
            Pushed += 'A'
        if is_pressed('s'):
            Pushed += 'S'
        if is_pressed('d'):
            Pushed += 'D'

        if Pushed == '':
            Pushed += 'N'

        if Pushed in Dic.keys():
            # print(Pushed)
            filename = folder + str(Pushed) + str(Dic[Pushed]) + ".jpg"
            # 截图
            # window_capture(filename)
            img = pyautogui.screenshot(region=[0, 0, 640, 480])  # x,y,w,h
            img.save(filename)
            # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            # CThread = threading.Thread(target=ChangeSize, args=(filename, ))
            # CThread.start()
            Dic[Pushed] += 1
            time.sleep(0.05)
            # print(time.time())

        if is_pressed('e'):
            break

        # print(time.asctime(time.localtime(time.time())))
