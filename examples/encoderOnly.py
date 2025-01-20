import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# ========== 1. 数据准备 ==========
class DummyDataset(Dataset):
    def __init__(self, size=1000, seq_len=10, vocab_size=50):
        self.data = torch.randint(1, vocab_size, (size, seq_len))  # 随机生成数据
        self.labels = torch.randint(0, 2, (size,))  # 修改为分类任务的标签

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

dataset = DummyDataset()
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ========== 2. 定义Encoder-only Transformer模型 ==========
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)  # 输出层改为分类

    def forward(self, src):
        # 添加位置编码
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        src = src.permute(1, 0, 2)  # (seq_len, batch, feature)
        
        # 通过encoder
        output = self.transformer_encoder(src)
        
        # 使用序列的最后一个时间步的输出进行分类
        output = output[-1]  # 取最后一个时间步
        output = self.fc_out(output)  # 分类预测
        return output

# ========== 3. 训练与评估 ==========
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, labels in data_loader:
        src, labels = src.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for src, labels in data_loader:
            src, labels = src.to(device), labels.to(device)
            output = model(src)
            loss = criterion(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return total_loss / len(data_loader), accuracy

# ========== 4. 主函数 ==========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50
    num_classes = 2
    model = EncoderOnlyTransformer(vocab_size, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        train_loss = train(model, data_loader, optimizer, criterion, device)
        val_loss, accuracy = evaluate(model, data_loader, criterion, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
