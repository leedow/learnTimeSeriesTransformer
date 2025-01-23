import torch
import torch.nn as nn

# 创建一个嵌入层，输入维度为10，输出维度为3
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)

# 查看嵌入层的权重矩阵
print("Embedding weights before training:")
print(embedding.weight)

# 创建一个输入张量，包含索引
input_indices = torch.tensor([1, 2, 3, 4])

# 获取嵌入向量
embedding_vectors = embedding(input_indices)

print("Embedding vectors:")
print(embedding_vectors)
