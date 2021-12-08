# coding=utf-8
"""
@author: Cao Zhanxiang
@project: PreTest
@file: Drive.py
@date: 2021/12/7
@function: 使用模型
"""
import time
import torch
import pyautogui
from torch import nn
from PIL import Image
from torchvision import transforms as F
from threading import Thread
from pykeyboard import PyKeyboard
from keyboard import *

# label to key dict
Dict = {
    'W': 0,
    'A': 1,
    'S': 2,
    'D': 3,
    'WA': 4,
    'WD': 5,
    'AS': 6,
    'SD': 7,
    'N': 8,
    0: 'W',
    1: 'A',
    2: 'S',
    3: 'D',
    4: 'WA',
    5: 'WD',
    6: 'AS',
    7: 'SD',
    8: ''
}

# net
class DriveNet(nn.Module):
    def __init__(self):
        super(DriveNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 9),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(-1, 512 * 4 * 4)
        out = self.fc(out)
        return out

class PredThread(Thread):
    def __init__(self, net):
        super(PredThread, self).__init__()
        self.DoRun = True
        self.net = net
        # 用''代替'N'表示None
        self.press = ''

    def run(self):
        """Run"""
        while self.DoRun:
            img = pyautogui.screenshot(region=[0, 0, 640, 480])  # x,y,w,h
            # 640 x 480 -> 640 x 640
            img = process_each_pic(img)
            img = F.ToTensor()(img)
            # 1 x 640 x 640
            img = torch.unsqueeze(img, 0)
            output = self.net(img)
            pred = torch.argmax(output, 1)
            press = Dict[pred.item()]
            self.press = press
    
    def GetPress(self):
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


if __name__ == "__main__":
    net = torch.load("./net0.pkl", map_location=torch.device('cpu'))
    # screen shoot and predict
    PThread = PredThread(net)

    k = PyKeyboard()
    current = ''
    # press P to start
    while True:
        if is_pressed('p'):
            break

    PThread.start()
    # press O to stop
    while True:
        temp = PThread.GetPress()
        if temp != current:
            for ch in current:
                k.release_key(ch)
            for ch in temp:
                k.press_keys(ch)
            current = temp
        print(temp)
        if is_pressed('o'):
            print('end')
            PThread.Stop()
            break
