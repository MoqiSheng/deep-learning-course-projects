# 案例包 · 短时交通流量预测

一个路段每 15 分钟产生一次车辆计数。站在当前时刻，使用过去四个完整时段，预测下一时段的车流量。本例使用固定随机种子生成的教学数据，包含工作日与周末的早晚高峰。

## 复现实验

在本目录执行以下命令。参考环境为 Python 3、NumPy 和 CPU 版 PyTorch 2.7.1。

```sh
python -m pip install numpy torch
python baseline.py
python train_gru.py
```

## 数据与划分

| 文件 | 内容 |
| --- | --- |
| [synthetic-flow.csv](synthetic-flow.csv) | 21 天、2,016 条模拟记录。interval_start 为时段起点，vehicles 为随后 15 分钟内的车辆数。 |
| [test-predictions.csv](test-predictions.csv) | 最后三天的模拟观测、上一时段预测和四步滞后线性回归预测。 |
| [gru-predictions.csv](gru-predictions.csv) | 对应 288 个时段的 GRU 预测，可按时间与前表连接。 |

按时间顺序取前 14 天训练、随后 4 天验证、最后 3 天测试。每次预测使用当时已经观测到的四个值；这是滚动单步预测。GRU 有 16 个隐藏单元，根据验证集平均绝对误差选取训练轮次。三个模型使用相同测试区间。

## 课堂讨论

- 从测试集中找出实际峰值。峰值前后哪个模型的误差最大？把误差的方向和大小分别画出来。
- 将历史窗口改为八个时段，保持数据划分一致。窗口变长后，预测是否更容易滞后？
- 如果部署时某个检测器连续缺测半小时，哪些输入必须替换？设计只利用过去数据的填补方法，并报告影响。

真实数据拓展可用 UCI 的小时级车流量数据，按小时重新定义窗口与目标。不要将不同采样频率的指标直接比较。[UCI 数据说明](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
