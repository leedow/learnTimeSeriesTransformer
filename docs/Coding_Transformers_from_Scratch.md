# 从零开始编写Transformer

在这个教程中，我们将使用PyTorch+Lightning构建和运行如图所示结构的Decoder-only transformer。Decoder-Only Transformers因在chatgpt中的应用而闻名。

图

尽管Decoder-Only Transformers 看起来很复杂，而且可以做很酷的事情，但幸运的是他的实现并不需要很多代码，很多功能我们只需从已有的组件中构建。因此在本教程中，你将实现以下事情：

从零开始编写一个位置编码器类（position encoder class）:位置编码器为transformer提供对输入的tokens位置跟踪的能力
从零开始编写一个注意力类（attention class）:注意力类为transformer提供分析输入和输入关系的能力
从零开始编写纯解码变形金钢模型（decoder-only transformer）:decoder-only transformer会整合我们基于pytorch编写位置编码和注意力类，实现输入输出功能。
训练模型：我们将训练模型回答简单的问题
使用训练好的模型：最后我们将使用模型回答简单的问题


注意：
本教程默认你会使用python，熟悉Decoder-Only Transformers和Backpropagation（反向传播）背后的理论。同时熟悉神经网络中的矩阵知识。如果你不熟悉以上内容，可以通过链接去学习。

强烈建议：
尝试运行代码，可以帮助你更好的学习理解它

## 引入所有的依赖

第一件事情是引用所有依赖。python只是一个编程语言，这些模块为我们提供了构建模型的额外功能。

注意：以下代码将检查lightning是否安装，如果没有将自动安装。同时你也需要安装pytorch。

```
## 检查lightning是否安装，没有则自动安装
import pip
try:
  __import__("lightning")
except ImportError:
  pip.main(['install', "lightning"])  

import torch ## torch可以创建张量以及提供基本的辅助函数
import torch.nn as nn ## torch.nn 提供了us nn.Module(), nn.Embedding() and nn.Linear()
import torch.nn.functional as F # 提供了softmax() and argmax()
from torch.optim import Adam ## 我们将使用Adam优化器, which is, essentially, 
                             ## a slightly less stochastic version of stochastic gradient descent.
from torch.utils.data import TensorDataset, DataLoader ## We'll store our data in DataLoaders

import lightning as L ## Lightning使编写代码更简单
```

## 创建输入输出数据集
本教程中我们将构建一个Decoder-Only Transformer模型可以回答两个简单的问题：What is StatQuest?和StatQuest is what?同时回答相同的答案：Awesome!!!

为了追踪我们简单的数据集，我们将创建一个字典，将单词和tokens映射到ID数字。因为我们即将使用的nn.Embedding()方法只接受id作为输入而不是文本。然后我们将使用这个字典创建一个Dataloader，其中包含了问题和预期的答案（以ID的形式）。最终我们使用这个dataloader训练模型。注意：dataloader被设计成处理成大规模数据，因此这个简单的例子依旧适用于大量数据的情况。

同时注意：输入和标签的训练数据可能看起来有点奇怪，因为decoder-only transformer会生成很多用户输入数据附加到输出上。为了进一步理解，我们假设我们想训练模型回答问题：what is statquest？答案为awesome。在左下图，我们可以看到第一个输入的token生成了输出is。在训练过程中，我们可以将这个输出对比已知的第二个输入，如果不相同，使用该损失修正模型中的权重和偏移参数。因此尽管is是输入的一部分，它同时也是标签的一部分，我们使用这个标签计算损失并决定是否需要优化参数。另一方面，statquest,<\eos>和awesome同样可以是输入和标签数据因为decoder-only transformer会通过输入的数据生成这些输出。

```
## 先创建一个字段，将输入token映射到id
token_to_id = {'what' : 0,
               'is' : 1,
               'statquest' : 2,
               'awesome': 3,
               '<EOS>' : 4, ## <EOS> = end of sequence
              }
## 然后创建一个字典将id映射到token,用于生成输出
## 我们使用map方法
## in the token_to_id dictionary. We then use dict() to make a new dictionary from the
## reversed tuples.
id_to_token = dict(map(reversed, token_to_id.items()))

## NOTE: Because we are using a Decoder-Only Transformer, the inputs contain
##       the questions ("what is statquest?" and "statquest is what?") followed
##       by an <EOS> token followed by the response, "awesome".
##       This is because all of those tokens will be used as inputs to the Decoder-Only
##       Transformer during Training. (See the illustration above for more details) 
## ALSO NOTE: When we train this way, it's called "teacher forcing".
##       Teacher forcing helps us train the neural network faster.
inputs = torch.tensor([[token_to_id["what"], ## input #1: what is statquest <EOS> awesome
                        token_to_id["is"], 
                        token_to_id["statquest"], 
                        token_to_id["<EOS>"],
                        token_to_id["awesome"]], 
                       
                       [token_to_id["statquest"], # input #2: statquest is what <EOS> awesome
                        token_to_id["is"], 
                        token_to_id["what"], 
                        token_to_id["<EOS>"], 
                        token_to_id["awesome"]]])

## NOTE: Because we are using a Decoder-Only Transformer the outputs, or
##       the predictions, are the input questions (minus the first word) followed by 
##       <EOS> awesome <EOS>.  The first <EOS> means we're done processing the input question
##       and the second <EOS> means we are done generating the output.
##       See the illustration above for more details.
labels = torch.tensor([[token_to_id["is"], 
                        token_to_id["statquest"], 
                        token_to_id["<EOS>"], 
                        token_to_id["awesome"], 
                        token_to_id["<EOS>"]],  
                       
                       [token_to_id["is"], 
                        token_to_id["what"], 
                        token_to_id["<EOS>"], 
                        token_to_id["awesome"], 
                        token_to_id["<EOS>"]]])

## Now let's package everything up into a DataLoader...
dataset = TensorDataset(inputs, labels) 
dataloader = DataLoader(dataset)
```

## 位置编码
位置编码帮助transformer追踪输入和输出单词的位置。比如在以下图片中我们看到两个词组squatch eats pizza和pizza eats squatch有着相同的单词，但是因为单词的顺序不同所以含义是不一样的。因此跟踪单词的顺序十分重要。

有很多方法可以实现单词顺序的追踪，但是一个流行的方法是使用sin,cose曲线方法。sine和cosine的数量取决于有多少的数字或者词嵌入的值，我们用来代表每个token。


如我们看到的图表，额外增加的sine和cosine相比之前的曲线有更长的周期，循环的次数更少。增加曲线的周期可以确保每一个位置可以代表唯一的取值组合。

注意：我们创建一个位置编码类的原因是这个类可以同时在encoder-only或者encoder-decoder transformer中重复利用，而不是直接把相关代码加到tarnsformer中。通过编写这些代码我们可以做到开发一次，在需要的时候随时调用他们。

同时注意：因为位置编码是固定不变的，意味着无论第一个token的值是什么他们总是使用相同的位置编码，我们预先计算好他们并且保存到一个表格中。这样使得添加位置编码变得更加高效。

现在我们理解了位置编码类的核心思想，开始编程：

```
class PositionEncoding(nn.Module):
    
    def __init__(self, d_model=2, max_len=6):
        ## d_model = 定义了transformer的维度，即代表了每一个输入token的词嵌入维度 
        ## max_len = 允许的输入的token最大长度。因为我们会预先计算好位置编码信息并且将他们保存到表格。我们可以使用d_model和max_len定义这个表格的行列的数量。在这个简单的例子中我们只使用简短的词汇，所以我们设置max_len=6作为默认设置。但是在The Annotated Transformer中这个值为5000。
        
        super().__init__()
        ## We call the super's init because by creating our own __init__() method, we overwrite the one
        ## we inherited from nn.Module. So we have to explicity call nn.Module's __init__(), otherwise it
        ## won't get initialized. NOTE: If we didn't write our own __init__(), then we would not have
        ## to call super().__init__(). Alternatively, if we didn't want to access any of nn.Module's methods, 
        ## we wouldn't have to call it then either.


        ## Now we create a lookup table, pe, of position encoding values and initialize all of them to 0.
        ## To do this, we will make a matrix of 0s that has max_len rows and d_model columns.
        ## for example...
        ## torch.zeros(3, 2)
        ## ...returns a matrix of 0s with 3 rows and 2 columns...
        ## tensor([[0., 0.],
        ##         [0., 0.],
        ##         [0., 0.]])


        ## 创建一个存储所有位置编码的表格pe,将值全部初始化为0
        ## 我们创建一个max_len行，d_model列的，值为0的矩阵
        pe = torch.zeros(max_len, d_model)

        ## Now we create a sequence of numbers for each position that a token can have in the input (or output).
        ## For example, if the input tokens where "I'm happy today!", then "I'm" would get the first
        ## position, 0, "happy" would get the second position, 1, and "today!" would get the third position, 2.
        ## NOTE: Since we are going to be doing math with these position indices to create the 
        ## positional encoding for each one, we need them to be floats rather than ints.
        ## 
        ## NOTE: Two ways to create floats are...
        ##
        ## torch.arange(start=0, end=3, step=1, dtype=torch.float)
        ##
        ## ...and...
        ##
        ## torch.arange(start=0, end=3, step=1).float()
        ##
        ## ...but the latter is just as clear and requires less typing.
        ##
        ## Lastly, .unsqueeze(1) converts the single list of numbers that torch.arange creates into a matrix with
        ## one row for each index, and all of the indices in a single column. So if "max_len" = 3, then we
        ## would create a matrix with 3 rows and 1 column like this...
        ##
        ## torch.arange(start=0, end=3, step=1, dtype=torch.float).unsqueeze(1)
        ##
        ## ...returns...
        ##
        ## tensor([[0.],
        ##         [1.],
        ##         [2.]])        

        ## 现在创建每一个token的位置数字序列
        ## 比如，如果输入tokens I'm happy today!,I'm作为第一个位置编码值为0,happy是第二个位置编码1,today!第三个位置编码。
        ## 注意：因为我们将使用数学计算每一个位置编码，因为使用floats数据类型而不是int
        ## 注意：有两种方法创建floats类型：
        ## torch.arange(start=0, end=3, step=1, dtype=torch.float)
        ## 以及
        ## torch.arange(start=0, end=3, step=1).float()
        ## 第二种更清晰，代码更简洁

        ## 最后.unsqueeze(1)可以将torch.arrange生成元素增加一个维度
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)


        ## Here is where we start doing the math to determine the y-axis coordinates on the
        ## sine and cosine curves.

        ## 开始使用sine和cosine曲线确定y值
        ##
        ## The positional encoding equations used in "Attention is all you need" are...

        ## 位置编码公式如下：
        ##
        ## PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        ## PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        ##
        ## ...and we see, within the sin() and cos() functions, we divide "pos" by some number that depends
        ## on the index (i) and total number of PE values we want per token (d_model). 

        ## 我们看到，通过sin,cos函数，我们根据索引i以及期望的的d_model值划分了一些数字
        ##
        ## NOTE: When the index, i, is 0 then we are calculating the y-axis coordinates on the **first pair** 
        ##       of sine and cosine curves. When i=1, then we are calculating the y-axis coordiantes on the 
        ##       **second pair** of sine and cosine curves. etc. etc.

        ## 注意：索引i为0时，我们计算对应sin,cose的Y值，i等于1时，再次计算sin,cos的y值
        ##
        ## Now, pretty much everyone calculates the term we use to divide "pos" by first, and they do it with
        ## code that looks like this...

        ## 现在，
        ##
        ## div_term = torch.exp(torch.arange(start=0, end=d_model, step=2).float() * -(math.log(10000.0) / d_model))
        ##
        ## Now, at least to me, it's not obvious that div_term = 1/(10000^(2i/d_model)) for a few reasons:
        ##
        ##    1) div_term wraps everything in a call to torch.exp() 
        ##    2) It uses log()
        ##    2) The order of the terms is different 
        ##
        ## The reason for these differences is, presumably, trying to prevent underflow (getting too close to 0).
        ## So, to show that div_term = 1/(10000^(2i/d_model))...
        ##
        ## 1) Swap out math.log() for torch.log() (doing this requires converting 10000.0 to a tensor, which is my
        ##    guess for why they used math.log() instead of torch.log())...
        ##
        ## torch.exp(torch.arange(start=0, end=d_model, step=2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        ##
        ## 2) Rearrange the terms...
        ##
        ## torch.exp(-1 * (torch.log(torch.tensor(10000.0)) * torch.arange(start=0, end=d_model, step=2).float() / d_model))
        ##
        ## 3) Pull out the -1 with exp(-1 * x) = 1/exp(x)
        ##
        ## 1/torch.exp(torch.log(torch.tensor(10000.0)) * torch.arange(start=0, end=d_model, step=2).float() / d_model)
        ##
        ## 4) Use exp(a * b) = exp(a)^b to pull out the 2i/d_model term...
        ##
        ## 1/torch.exp(torch.log(torch.tensor(10000.0)))^(torch.arange(start=0, end=d_model, step=2).float() / d_model)
        ##
        ## 5) Use exp(log(x)) = x to get the original form of the denominator...
        ##
        ## 1/(torch.tensor(10000.0)^(torch.arange(start=0, end=d_model, step=2).float() / d_model))
        ##
        ## 6) Bam.
        ## 
        ## So, that being said, I don't think underflow is actually that big an issue. In fact, some coder at Hugging Face
        ## also doesn't think so, and their code for positional encoding in DistilBERT (a streamlined version of BERT, which
        ## is a transformer model)
        ## calculates the values directly - using the form of the equation found in original Attention is all you need
        ## manuscript. See...
        ## https://github.com/huggingface/transformers/blob/455c6390938a5c737fa63e78396cedae41e4e87e/src/transformers/modeling_distilbert.py#L53
        ## So I think we can simplify the code, but I'm also writing all these comments to show that it is equivalent to what
        ## you'll see in the wild...
        ##
        ## Now let's create an index for the embedding positions to simplify the code a little more...
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        ## NOTE: Setting step=2 results in the same sequence numbers that we would get if we multiplied i by 2.
        ##       So we can save ourselves a little math by just setting step=2.

        ## And now, finally, let's create div_term...
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        
        ## Now we calculate the actual positional encoding values. Remember 'pe' was initialized as a matrix of 0s
        ## with max_len (max number of input tokens) rows and d_model (number of embedding values per token) columns.
        pe[:, 0::2] = torch.sin(position * div_term) ## every other column, starting with the 1st, has sin() values
        pe[:, 1::2] = torch.cos(position * div_term) ## every other column, starting with the 2nd, has cos() values
        ## NOTE: If the notation for indexing 'pe[]' looks cryptic to you, read on...
        ##
        ## First, let's look at the general indexing notation:
        ##
        ## For each row or column in matrix we can select elements in that
        ## row or column with the following indexs...
        ##
        ## i:j:k = select elements between i and j with stepsize = k.
        ##
        ## ...where...
        ##
        ## i defaults to 0
        ## j defaults to the number of elements in the row, column or whatever.
        ## k defaults to 1
        ##
        ## Now that we have looked at the general notation, let's look at specific
        ## examples so that we can understand it.
        ##
        ## We'll start with: pe[:, 0::2]
        ##
        ## The stuff that comes before the comma (in this case ':') refers to the rows we want to select.
        ## The ':' before the comma means "select all rows" because we are not providing specific 
        ## values for i, j and k and, instead, just using the default values.
        ##
        ## The stuff after the comma refers to the columns we want to select.
        ## In this case, we have '0::2', and that means we start with
        ## the first column (column =  0) and go to the end (using the default value for j)
        ## and we set the stepsize to 2, which means we skip every other column.
        ##
        ## Now to understand pe[:, 1::2]
        ##
        ## Again, the stuff before the comma refers to the rows, and, just like before
        ## we use default values for i,j and k, so we select all rows.
        ##
        ## The stuff that comes after the comma refers to the columns.
        ## In this case, we start with the 2nd column (column = 1), and go to the end
        ## (using the default value for 'j') and we set the stepsize to 2, which
        ## means we skip every other column.
        ##
        ## NOTE: using this ':' based notation is called "indexing" and also called "slicing"
        
        ## Now we "register 'pe'.
        self.register_buffer('pe', pe) ## "register_buffer()" ensures that
                                       ## 'pe' will be moved to wherever the model gets
                                       ## moved to. So if the model is moved to a GPU, then,
                                       ## even though we don't need to optimize 'pe', it will 
                                       ## also be moved to that GPU. This, in turn, means
                                       ## that accessing 'pe' will be relatively fast copared
                                       ## to having a GPU have to get the data from a CPU.

    ## Because this class, PositionEncoding, inherits from nn.Module, the forward() method 
    ## is called by default when we use a PositionEncoding() object.
    ## In other words, after we create a PositionEncoding() object, pe = PositionEncoding(),
    ## then pe(word_embeddings) will call forward() and so this is where 
    ## we will add the position encoding values to the word embedding values
    def forward(self, word_embeddings):
    
        return word_embeddings + self.pe[:word_embeddings.size(0), :] ## word_embeddings.size(0) = number of embeddings
                                                                      ## NOTE: That second ':' is optional and 
                                                                      ## we could re-write it like this: 
                                                                      ## self.pe[:word_embeddings.size(0)]
```


First, the general equations for the different types of attention are almost identical as seen in the figure below. In the equations, Q is for the Query matrix, K is for the Key matrix and V is for the Value matrix. On the left, we have the equation for Self-Attention and Encoder-Decoder Attention. The differences in these types of attention are not from the equation we use, but from how the Q, K, and V matrices are computed. On the right, we see the equation for Masked Self-Attention and the only difference it has from the equation on the left is the addition of a Mask matrix, M, that prevents words that come after a specific Query from being included in the final attention scores.
Equations for computing attention

一般来说，不同类型注意力的公式几乎相同，如下图所示。在公式中，Q表示查询矩阵（Query matrix），K表示键矩阵（Key matrix），V 表示值矩阵（Value matrix）。左侧显示的是自注意力（Self-Attention）和编码器-解码器注意力（Encoder-Decoder Attention）的公式。这些注意力类型的区别并不在于使用的公式，而在于如何计算 Q、K 和 V 矩阵。

右侧显示的是掩码自注意力（Masked Self-Attention）的公式，它与左侧的唯一区别在于增加了一个掩码矩阵 MM。这个掩码矩阵的作用是阻止特定查询之后的单词被包含在最终的注意力分数中。


NOTE: Since both equations are very similar, we'll go through one example and point out the key differences when we get to them.

注意：因为这些公式十分相似，我们将通过一个例子进行讲解，并在涉及到关键区别时加以说明。



First, given word embedding values for each word/token in the input phrase <SOS> let's go in matrix form, we multiply them by matrices of weights to create Queries, Keys, and Values

首先，给定输入短语中每个单词/标记的词嵌入值（如 <SOS>），将其以矩阵形式表示，我们将它们与权重矩阵相乘，以生成查询 (Queries)、键 (Keys) 和值 (Values)。


We then multiply the Queries by the transpose of the Keys so that the query for each word calculates a similarity value with the keys for all of the words. NOTE: As seen in the illustration below, Masked Self-Attention calculates the values for all Query/Key pairs, but, ultimately, ignores values for when a token's Query comes before other token's Keys. For example, if the Query is for the first token <SOS>, then Masked Self-Attention will ignore the values calculated with Keys for Let's and go because those tokens come after <SOS>. 


接着，我们将查询 (Queries) 矩阵与键 (Keys) 矩阵的转置相乘，这样每个单词的查询都会计算出与所有单词的键的相似值。注意：如下面的图示所示，掩码自注意力 (Masked Self-Attention) 会计算所有查询/键对的值，但最终会忽略某些情况下的值。例如，如果查询是针对第一个标记 <SOS>，那么掩码自注意力会忽略与 "Let's" 和 "go" 的键计算出的值，因为这些标记位于 <SOS> 之后。



The next step is to scale the similarity scores by the square root of the number of columns in the Key matrix, which represents the number of values used to represent each token. In this case, we scale by the square root of 2


下一步是将相似度分数按照键 (Key) 矩阵列数的平方根进行缩放，这个列数表示用于表示每个标记的值的数量。在这个例子中，我们将相似度分数除以根号2


Now, if we were doing Masked Self-Attention, we would mask out the values we want to ignore by adding -infinity to them, as seen below. This step is the only difference between Self-Attention and Masked Self-Attention. 

现在，如果我们在进行 Masked Self-Attention（掩蔽自注意力） 操作，我们会通过将负无穷大（-∞）添加到我们想忽略的值上来屏蔽它们，如下图所示。这一步是 Self-Attention（自注意力） 和 Masked Self-Attention 之间的唯一区别。






```
class Attention(nn.Module): 
    
    def __init__(self, d_model=2):
        ## d_model = embedding 的数量
        ##           In the transformer I used in the StatQuest: Transformer Neural Networks Clearly Explained!!!
        ##           d_model=2, so that's what we'll use as a default for now.
        ##           However, in "Attention Is All You Need" d_model=512

        
        super().__init__()
        
        self.d_model=d_model
        
        ## Initialize the Weights (W) that we'll use to create the
        ## query (q), key (k) and value (v) numbers for each token

        ## 初始化权重W，我们将使用w去为每个token创建q,k,V

        ## NOTE: Most implementations that I looked at include the bias terms
        ##       but I didn't use them in my video (since they are not in the 
        ##       original Attention is All You Need paper).


        ## 注意：大多数实现中我看到都包括了偏置项（bias terms），但我在我的视频中没有使用它们（因为它们并未出现在原始的《Attention is All You Need》论文中）。

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        
        ## NOTE: In this simple example, we are not training on the data in "batches"
        ##       However, by defining variables for row_dim and col_dim, we could
        ##       allow for batches by setting row_dim to 1 and col_com to 2.

        ## 在这个简单的示例中，我们并未以“批次”（batches）的方式对数据进行训练。
然而，通过定义用于行维度（row_dim）和列维度（col_dim）的变量，我们可以通过将row_dim设置为1和col_dim设置为2来实现批量处理的功能。
        self.row_dim = 0
        self.col_dim = 1

        
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        ## Create the query, key and values using the encodings
        ## associated with each token (token encodings)
        ##
        ## NOTE: For Encoder-Decoder Attention, the encodings for q come from
        ##       the decoder and the encodings for k and v come from the output
        ##       from the encoder.
        ##       In all of the other types of attention, the encodings all
        ##       come from the same source.
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        ## Compute attention scores
        ## the equation is (q * k^T)/sqrt(d_model)
        ## NOTE: It seems most people use "reverse indexing" for the dimensions when transposing k
        ##       k.transpose(dim0, dim1) will transpose k by swapping dim0 and dim1
        ##       In standard matrix notation, we would want to swap rows (dim=0) with columns (dim=1)
        ##       If we have 3 dimensions, because of batching, and the batch was the first dimension
        ##       And thus dims are defined batch = 0, rows = 1, columns = 2
        ##       then dim0=-2 = 3 - 2 = 1. dim1=-1 = 3 - 1 = 2.
        ##       Alternatively, we could put the batches in dim 3, and thus, dim 0 would still be rows
        ##       and dim 1 would still be columns. I'm not sure why batches are put in dim 0...
        ##
        ##       Likewise, the q.size(-1) uses negative indexing to rever to the number of columns in the query
        ##       which tells us d_model. Alternatively, we could ust q.size(2) if we have batches in the first
        ##       dimension or q.size(1) if we have batches in the 3rd dimension.
        ##
        ##       Since there are a bunch of ways to index things, I think the best thing to do is use
        ##       variables "row_dim" and "col_dim" instead of numbers...
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            ## Here we are masking out things we don't want to pay attention to,
            ## like tokens that come after the current token.
            ## We can also use masking to block out the <PAD> token,
            ## which is used when we have a batch of inputs sequences
            ## and they are not all the exact same length. Because the batch is passed
            ## in as a matrix, each input sequence has to have the same length, so we
            ## add <PAD> to the shorter sequences so that they are all as long ast the
            ## longest sequence.
            ##
            ## We replace <PAD>, or tokens that come after the current token
            ## with a very large negative number so that the SoftMax() function
            ## will give all masked elements an output value (or "probability") of 0.
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9) # I've also seen -1e20 and -9e15 used in masking
        
        ## Apply softmax to determine what percent of each token's value to
        ## use in the final attention values.
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        ## Scale the values by their associated percentages and add them up.
        attention_scores = torch.matmul(attention_percents, v)
        
        return attention_scores
```