# NeedForSpeedAI
the AI to drive the car in the game Need For Speed

---

### 文件结构

├─App	模型应用
│  ├─save	模型存储
├─Data	获取数据以及数据集制作
│  ├─Collect	获取数据脚本
│  │  ├─dataset	制作的数据集
│  │  │  ├─dataset_czx
│  │  │  ├─dataset_mcl
│  │  │  └─dataset_xyx
│  │  ├─data_raw	原始截图文件
│  │  │  ├─czx1
│  │  │  ├─mcl1
│  │  │  │  └─data	分割的json文件
│  │  │  └─xyx1
│  │  │      └─data	分割的json文件
│  ├─dataset	副本
│  ├─nfs	制作好的数据集
│  │  ├─test	测试集原图
│  │  ├─test_ann	测试集分割结果
│  │  ├─train	训练集原图
│  │  ├─train_ann	训练集分割结果
│  │  ├─val	验证集原图
│  │  └─val_ann	验证集分割结果
│  ├─root	临时
│  └─root_ann	临时
├─DriveNet	网络训练
│  └─save	保存的模型
└─ENet	网络训练
    ├─data	
    ├─metric
    ├─models
    └─save
        ├─ENet_CamVid
        └─ENet_Cityscapes

---

### to_dataset.sh的使用

​		语义分割数据集标注后通过此脚本将.json格式标注转化为数据集

+ 使用方法：可以使用git的shell解释器，对某个文件夹下的.json文件批量操作

	```shell
	    sh ./to_dataset.sh [JSON_FOLDER_NAME] [OUT_FOLDER_NAME]
	```

---

### pywinio环境搭建

​		参见[python库 pywinio环境搭建](https://www.cnblogs.com/chenjy1225/p/12162505.html)

>##### Note:
>
>+ 进入[启动设置](https://www.cnblogs.com/chenjy1225/p/12162505.html#:~:text=%E5%AE%8C%E6%88%90%E5%AE%89%E8%A3%85%E5%8D%B3%E5%8F%AF%E3%80%82-,%E7%A6%81%E7%94%A8%E9%A9%B1%E5%8A%A8%E7%A8%8B%E5%BA%8F%E5%BC%BA%E5%88%B6%E7%AD%BE%E5%90%8D,-%E9%87%8D%E5%90%AFf8)可从设置->系统->恢复->高级启动->立即重新启动
>
>+ 启动设置`禁用驱动程序强制签名`时，需要BitLocker恢复密钥，登录[Microsoft](https://account.microsoft.com/devices/recoverykey?refd=support.microsoft.com)查询
>+ 运行python脚本时，需要以**管理员权限**运行PyCharm或其他

---

### rename.bat脚本的使用

​		用于对同一归属目录下的文件夹统一重命名，在其后加上**_[数字]**，具体数字需编辑更改，慎用

