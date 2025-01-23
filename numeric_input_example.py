import torch
import torch.nn as nn

# 定义一个简单的线性层
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 3)  # 输入维度为1，输出维度为3

    def forward(self, x):
        return self.linear(x)

# 创建模型实例
model = SimpleModel()

# 创建一个输入张量，包含纯数字
input_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

# 获取模型输出
output = model(input_data)

print(output)
