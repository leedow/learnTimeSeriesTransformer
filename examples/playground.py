import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

d_model = 128
seq_len = 5
vocab_size = 50

#print(torch.zeros(1,2,5))
#print(nn.Parameter(torch.zeros(1,2,5)))

test = torch.randint(1,10,(3,5,2))
print(test)

print(test.permute(1,0,2))


class datas(Dataset):
    def __init__(self):
        self.data = torch.randint(1,vocab_size,(10,seq_len))
        self.lables = torch.randint(1,10,(10,))

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)
    

class transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1,seq_len,d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=4,dim_feedforward=512)

        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=2)

        self.out = nn.Linear(d_model,10)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]