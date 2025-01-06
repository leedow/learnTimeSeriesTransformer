import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# ========== 1. 数据准备 ==========
class DummyDataset(Dataset):
    def __init__(self, size=1000, seq_len=10, vocab_size=50):
        self.data = torch.randint(1, vocab_size, (size, seq_len))  # 随机生成数据
        self.labels = torch.randint(0, vocab_size, (size, seq_len))

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

dataset = DummyDataset()
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ========== 2. 定义Transformer模型 ==========
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))  # 简单的固定位置编码
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, dim_feedforward=dim_feedforward
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]
        src, tgt = src.permute(1, 0, 2), tgt.permute(1, 0, 2)  # Transformer期望(seq_len, batch, feature)
        output = self.transformer(src, tgt)
        output = self.fc_out(output.permute(1, 0, 2))  # 返回(batch, seq_len, vocab_size)
        return output

# ========== 3. 训练与评估 ==========
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    for src, tgt in data_loader:
        src, tgt_input, tgt_output = src.to(device), tgt[:, :-1].to(device), tgt[:, 1:].to(device)
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt_input, tgt_output = src.to(device), tgt[:, :-1].to(device), tgt[:, 1:].to(device)
            output = model(src, tgt_input)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

# ========== 4. 主函数 ==========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50
    model = Transformer(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD=0的损失

    for epoch in range(5):  # 训练5个epoch
        train(model, data_loader, optimizer, criterion, device)
        val_loss = evaluate(model, data_loader, criterion, device)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()
