import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 1. 多头自注意力机制 =====================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadSelfAttention, self).__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每个头的维度
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, C = x.shape  # B: batch size, L: seq length, C: embedding dim (d_model)
        # 线性变换并分割为多个头
        Q = self.linear_q(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, L, d_k)
        K = self.linear_k(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        V = self.linear_v(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, nhead, L, L)
        attn = F.softmax(scores, dim=-1)

        # 加权求和
        context = torch.matmul(attn, V)  # (B, nhead, L, d_k)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)  # 合并多个头

        return self.fc_out(context)  # 最终输出

# ===================== 2. 前馈神经网络 =====================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# ===================== 3. Transformer 编码器层 =====================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead)
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(src)
        src = self.norm1(src + self.dropout1(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(src)
        src = self.norm2(src + self.dropout2(ffn_output))

        return src

# ===================== 4. Transformer 编码器 =====================
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, vocab_size, max_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))  # 可学习的位置编码
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        B, L = src.shape
        x = self.embedding(src) + self.pos_encoding[:, :L, :]  # 嵌入和位置编码相加
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# ===================== 5. 模型定义与测试 =====================
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, max_len=100):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, vocab_size, max_len)
        self.fc_out = nn.Linear(d_model, vocab_size)  # 输出到词汇表大小

    def forward(self, src):
        x = self.encoder(src)
        return self.fc_out(x)  # 输出最终预测

# ===================== 6. 训练和测试 =====================
if __name__ == "__main__":
    # 参数设置
    vocab_size = 1000  # 词汇表大小
    max_len = 50  # 序列长度
    batch_size = 16
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 512

    # 模型实例化
    model = TransformerModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len)

    # 输入数据 (batch_size, sequence_length)
    src = torch.randint(0, vocab_size, (batch_size, max_len))  # 随机生成输入数据

    # 前向传播
    output = model(src)
    print("输出形状:", output.shape)  # (batch_size, seq_len, vocab_size)



