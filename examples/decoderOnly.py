import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# ========== 1. 数据准备 ==========
class DummyDataset(Dataset):
    def __init__(self, size=1000, seq_len=10, vocab_size=50):
        self.data = torch.randint(1, vocab_size, (size, seq_len))  # 随机生成数据
        # 在decoder-only中，我们只需要一个序列
        self.labels = torch.roll(self.data, shifts=-1, dims=1)
        self.labels[:, -1] = 0  # 最后一个位置填充0

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

dataset = DummyDataset()
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ========== 2. 定义Decoder-only Transformer模型 ==========
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 创建因果注意力掩码
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)

        # 嵌入和位置编码
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)

        # decoder层，使用因果掩码
        output = self.decoder(x, x, tgt_mask=causal_mask)
        output = self.fc_out(output.permute(1, 0, 2))  # (batch, seq_len, vocab_size)
        return output

# ========== 3. 训练与评估 ==========
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for input_seq, target_seq in data_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in data_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output = model(input_seq)
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

# ========== 4. 主函数 ==========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50
    model = DecoderOnlyTransformer(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(5):
        train_loss = train(model, data_loader, optimizer, criterion, device)
        val_loss = evaluate(model, data_loader, criterion, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()
