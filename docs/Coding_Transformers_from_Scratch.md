# 从零开始编写Transformer

在这个教程中，我们将使用PyTorch+Lightning构建和运行如图所示结构的Decoder-only transformer。Decoder-Only Transformers因在chatgpt中的应用而闻名。

图

尽管Decoder-Only Transformers 看起来很复杂，而且可以做很酷的事情，但幸运的是他的实现并不需要很多代码，很多功能我们只需从已有的组件中构建。因此在本教程中，你将实现以下事情：

从零开始编写一个位置编码器类（position encoder class）:位置编码器为transformer提供对输入的tokens位置跟踪的能力
从零开始编写一个注意力类（attention class）:注意力类为transformer提供分析输入和输入关系的能力
从零开始编写纯解码变形金钢模型（decoder-only transformer）:decoder-only transformer会整合我们基于pytorch编写位置编码和注意力类，实现输入输出功能。
训练模型：我们将训练模型回答简单的问题
使用训练好的模型：最后我们将使用模型回答简单的问题


注意：
本教程默认你会使用python，熟悉Decoder-Only Transformers和Backpropagation（反向传播）背后的理论。同时熟悉神经网络中的矩阵知识。如果你不熟悉以上内容，可以通过链接去学习。

强烈建议：
尝试运行代码，可以帮助你更好的学习理解它

## 引入所有的依赖

第一件事情是引用所有依赖。python只是一个编程语言，这些模块为我们提供了构建模型的额外功能。

注意：以下代码将检查lightning是否安装，如果没有将自动安装。同时你也需要安装pytorch。

```
## 检查lightning是否安装，没有则自动安装
import pip
try:
  __import__("lightning")
except ImportError:
  pip.main(['install', "lightning"])  

import torch ## torch可以创建张量以及提供基本的辅助函数
import torch.nn as nn ## torch.nn 提供了us nn.Module(), nn.Embedding() and nn.Linear()
import torch.nn.functional as F # 提供了softmax() and argmax()
from torch.optim import Adam ## 我们将使用Adam优化器, which is, essentially, 
                             ## a slightly less stochastic version of stochastic gradient descent.
from torch.utils.data import TensorDataset, DataLoader ## We'll store our data in DataLoaders

import lightning as L ## Lightning使编写代码更简单
```

## 创建输入输出数据集
本教程中我们将构建一个Decoder-Only Transformer模型可以回答两个简单的问题：What is StatQuest?和StatQuest is what?同时回答相同的答案：Awesome!!!

为了追踪我们简单的数据集，我们将创建一个字典，将单词和tokens映射到ID数字。因为我们即将使用的nn.Embedding()方法只接受id作为输入而不是文本。然后我们将使用这个字典创建一个Dataloader，其中包含了问题和预期的答案（以ID的形式）。


