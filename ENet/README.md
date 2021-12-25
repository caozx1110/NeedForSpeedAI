### Usage

+ `ENet.ipynb` 训练网络

---

### 文件结构

+ `data/` 存放直接作为网络输入的训练集，验证集和测试集 
+ `dataset/` 存放标注完的各个样本的 $json$ 文件夹
+ `metric/`  分析 IoU 等网络表现指标
+ `models/` 存放 ENet 的网络结构
+ `save/` 用于存放模型 
+ `ENet.ipynb` 用于训练和测试 ENet 
+ `args.py` 网络输入指标
+ `data_process.py` 处理数据
+ `export.py` 将后续模型需要的输出外传的接口
+ `main.py` 运行训练或者测试过程
+ `test_.py` 构建测试模型
+ `train.py` 构建训练模型
+ `transforms.py` 数据转换
+ `utils.py` 其他的工具

