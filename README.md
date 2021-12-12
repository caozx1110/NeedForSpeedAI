# NeedForSpeedAI
the AI to drive the car in the game Need For Speed

---

### GetData.py

用于在游戏中获取玩家的数据

---

### Drive.py

用于根据训练好的模型预测按键输入，并驱动车

---

### KeyboardEmulation.py

模拟按键输入接口，使用winio库中接口进一步封装

---

### model.py

没啥用的网络结构

---

### to_dataset.sh

语义分割数据集标注后通过此脚本将.json格式标注转化为数据集

+ 使用方法：可以使用git的shell解释器，对某个文件夹下的.json文件批量操作
    ```shell
        sh ./to_dataset.sh [JSON_FOLDER_NAME] [OUT_FOLDER_NAME]
    ```

