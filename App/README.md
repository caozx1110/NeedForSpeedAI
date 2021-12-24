### Usage

+ 运行`Drive.py`进行预测和键盘模拟

---
### 文件结构

+ `save/`：保存的模型
    + aug0,1,2为3个版本ENet的模型
    + drive.pth及max.pth为DriveNet的模型

+ `Drive.py`：操控游戏

+ `DriveNet.py`：网络结构

+ `enet.py`：网络结构

+ `KeyboardEmulation.py`：pywinio的进一步封装

+ `ModelTest.py`：测试模型效果以及利用ENet生成DriveNet的数据集

+ `temp.png`：temp，测试用

+ `transforms.py`，`export.py`，`utils.py`：库函数

