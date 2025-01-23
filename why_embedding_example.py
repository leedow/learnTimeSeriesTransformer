import torch
import torch.nn as nn

# 假设我们有一个包含5个单词的词汇表
vocab_size = 5
embedding_dim = 3

# 创建一个嵌入层
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# 创建一个输入张量，包含单词索引
input_indices = torch.tensor([0, 1, 2, 3, 4])

# 获取嵌入向量
embedding_vectors = embedding(input_indices)

print("Embedding vectors:")
print(embedding_vectors)

# 假设我们使用 one-hot 编码表示这些单词
one_hot_vectors = torch.eye(vocab_size)

print("One-hot vectors:")
print(one_hot_vectors)
