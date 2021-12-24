# coding=utf-8
"""
@author: Cao Zhanxiang
@project: 8-Queens.py
@file: GetData.py
@date: 2021/11/14
@function: 获取原始数据
"""
import time

# import win32con
# import win32gui
# import win32ui
from keyboard import *
# from PIL import Image
import os
import pyautogui

# def ChangeSize(filename):
#     img = Image.open(filename)
#     change = img.resize((640, 480))
#     change.save(filename)


if __name__ == '__main__':
    # TODO: change the folder name
    folder = './data_raw/lph2/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    Dic = {'W': 0, 'A': 0, 'S': 0, 'D': 0, 'WA': 0, 'WD': 0, 'AS': 0, 'SD': 0, 'N': 0}
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
            Dic[Pushed] += 1
            # 间隔时间
            time.sleep(0.05)
            # print(time.time())

        if is_pressed('e'):
            break

        # print(time.asctime(time.localtime(time.time())))
