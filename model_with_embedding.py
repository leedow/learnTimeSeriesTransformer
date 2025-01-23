import torch
import torch.nn as nn

class ModelWithEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, output_dim):
        super(ModelWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x

# 创建模型实例
model = ModelWithEmbedding(num_embeddings=10, embedding_dim=3, output_dim=1)

# 创建一个输入张量，包含索引
input_indices = torch.tensor([1, 2, 3, 4])

# 获取模型输出
output = model(input_indices)

# 假设我们有一个查找表，将输出映射回原始类别
index_to_category = {0: 'cat', 1: 'dog', 2: 'bird', 3: 'fish', 4: 'horse'}

# 将输出映射回原始类别
predicted_categories = [index_to_category[idx.item()] for idx in input_indices]

print(predicted_categories)
