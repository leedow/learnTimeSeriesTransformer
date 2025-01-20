import torch
import torch.nn as nn
import torch.optim as optim
import math

# 定义Transformer模型类
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)  # 嵌入层
        self.pos_encoder = nn.Sequential(  # 位置编码器
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)  # 解码器层
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)  # Transformer解码器
        self.fc_out = nn.Linear(d_model, output_dim)  # 输出全连接层

    def forward(self, src, tgt, memory):
        src = self.embedding(src) * math.sqrt(self.d_model)  # 对源输入进行嵌入并缩放
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)  # 对目标输入进行嵌入并缩放
        src = self.pos_encoder(src)  # 对源输入进行位置编码
        tgt = self.pos_encoder(tgt)  # 对目标输入进行位置编码
        output = self.transformer_decoder(tgt, memory)  # 通过Transformer解码器
        output = self.fc_out(output)  # 通过全连接层输出
        return output

# 训练函数
def train(model, data, targets, criterion, optimizer, num_epochs):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # 清空梯度
        output = model(data, targets, data)  # 前向传播
        loss = criterion(output, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')  # 打印每个epoch的损失

# 预测函数
def predict(model, data):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        output = model(data, data, data)  # 前向传播
    return output

# 示例用法
input_dim = 10
output_dim = 10
d_model = 512
nhead = 8
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
num_epochs = 20

model = TransformerModel(input_dim, output_dim, d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data
data = torch.randint(0, input_dim, (10, 32))  # (sequence_length, batch_size)
targets = torch.randint(0, output_dim, (10, 32))

train(model, data, targets, criterion, optimizer, num_epochs)
predictions = predict(model, data)
print(predictions)