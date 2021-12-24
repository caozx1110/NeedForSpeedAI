### Usage

+ 使用`Collect/`目录下的脚本进行数据采集
+ 直接使用已标好的`drive/`、`nfs/`数据集

---

### 文件结构

+ `Collect/`：存放收集数据的脚本和原始数据
    + `data_raw/`：存放原始数据
    + `GetData.py`：获取数据
    + `ChangeLabel.py`：用于将误命名label为1，2的json文件改回来
    + `to_dataset.sh`：用于将labelme生成的json文件批量转换为可用数据集
    + `rename.bat`：用于将当前目录下的文件夹后添加数字后缀
+ `dataset/`：存放labelme转换出的数据集文件夹
+ `drive/`：用于DriveNet的训练的数据集
+ `nfs/`：用于语义分割ENet的数据集
+ `nfs_undivided/`：尚未分成训练验证测试集的ENet数据集