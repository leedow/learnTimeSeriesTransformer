import torch
import torch.nn as nn 

class SelfAttention(nn.Module):

	'''
	embed_size: 输入特征的维度（embedding 的大小），通常是一个固定值。
	heads:多头注意力的头数，多头的机制允许模型从不同的子空间学习特征。
	'''
	def __init__(self, embed_size, heads):
		super(SelfAttention, self).__init__()
		self.embed_size = embed_size
		self.heads = heads
		self.head_dim = embed_size // heads #每个头负责处理的特征维度大小,多头注意力机制会将 embed_size 分成 heads 个部分，每部分的大小为 embed_size // heads

		assert(self.head_dim *heads == embed_size), "embed size needs to be div by heads"

		'''
		Linear的基础用法：
			torch.nn.Linear(in_features, out_features, bias=True)
		参数说明：
			in_features：输入特征的维度，即每个输入样本的特征数
			out_features：输出特征的维度，即每个输出样本的特征数
			bias：是否包含偏置项，默认为 True
		输入形状通常为 (batch_size, in_features)
		'''
		self.values == nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.keys == nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.queries == nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.fc_count == nn.Linear(self.heads*self.head_dim, embed_size)

	def forward(self, values, keys, query, mask):
		N = query.shape[0]

		value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

		values = values.reshape(N, value_len, self.heads, self.head_dim)
		keys = keys.reshape(N, key_len, self.heads, self.head_dim)
		query = query.reshape(N, query_len, self.heads, self.head_dim)

		energy = torch.einsum()



