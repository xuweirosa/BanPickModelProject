### 1.环境配置 

- python3.7
- 代码在Windows和Mac上测试
- 依赖包：
  - numpy
  - pandas
  - tensorflow
  - sklearn
  - matplotlib
  - scipy

### 2.数据

以英雄联盟LOL职业比赛数据作为数据来源

- dataset.xlsx: Ban Pick决策模型使用的训练数据集

- data2.xlsx：实时胜率预测使用的训练数据集

### 3.代码

- BanPick决策模型部分，朴素贝叶斯代码为NaiveBayes_ban.py
- BanPick决策模型部分，深度神经网络代码为NN_ban.py
- BanPick决策模型部分，RNN实现代码为RNN.py
- 胜率预测代码为winning_rate.py
