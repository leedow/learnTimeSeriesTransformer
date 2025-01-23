import torch
import torch.nn as nn

# 定义模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]
        src = src.permute(1, 0, 2)  # (seq_len, batch, d_model)
        tgt = tgt.permute(1, 0, 2)  # (seq_len, batch, d_model)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.fc_out(output.permute(1, 0, 2))  # (batch, seq_len, vocab_size)
        return output

# 创建模型实例
vocab_size = 100
model = TransformerModel(vocab_size)

# 创建输入张量
src = torch.randint(0, vocab_size, (32, 10))  # (batch_size, seq_len)
tgt = torch.randint(0, vocab_size, (32, 10))  # (batch_size, seq_len)

# 获取模型输出
output = model(src, tgt)

print("Model output shape:", output.shape)
