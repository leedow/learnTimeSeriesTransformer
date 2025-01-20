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
        ## d_model = Transformer的维度，也就是每个token的嵌入值数量。
        ##           在我使用的StatQuest中的Transformer中，d_model=2，因此我们暂时使用这个作为默认值。
        ##           然而，在《Attention Is All You Need》中的d_model=512。
        ## max_len = 允许作为输入的最大token数量。
        ##           由于我们预计算了位置编码值并将它们存储在查找表中，
        ##           我们可以使用d_model和max_len来确定查找表中的行和列数。
        ##
        ##           在这个简单的示例中，我们只使用了短语，所以我们将max_len=6作为默认设置。
        ##           然而，在《The Annotated Transformer》中，他们将max_len的默认值设置为5000。
        
        super().__init__()
        ## 我们调用super的init方法，因为我们自己定义了__init__()方法，覆盖了从nn.Module继承的__init__()方法。
        ## 所以我们必须显式地调用nn.Module的__init__()，否则它不会被初始化。
        ## 注意：如果我们没有写自己的__init__()，那么就不需要调用super().__init__()。
        ## 另外，如果我们不打算访问nn.Module的任何方法，那么也不需要调用它。

        ## 现在我们创建一个位置编码值的查找表pe，并将所有值初始化为0。
        ## 为此，我们将创建一个0矩阵，具有max_len行和d_model列。
        ## 例如...
        ## torch.zeros(3, 2)
        ## ...返回一个3行2列的0矩阵...
        ## tensor([[0., 0.],
        ##         [0., 0.],
        ##         [0., 0.]])
        pe = torch.zeros(max_len, d_model)

        ## 现在我们为每个token在输入中的位置创建一个序列号（或者输出中的位置）。
        ## 例如，如果输入的tokens是"我今天很高兴！"，那么"我"会得到第一个位置0，
        ## "今天"会得到第二个位置1，"很高兴！"会得到第三个位置2。
        ## 注意：由于我们要对这些位置索引进行数学运算以创建每个位置的编码，
        ##       所以我们需要它们是浮点数，而不是整数。
        ## 
        ## 注意：创建浮点数的两种方式是...
        ##
        ## torch.arange(start=0, end=3, step=1, dtype=torch.float)
        ##
        ## ...和...
        ##
        ## torch.arange(start=0, end=3, step=1).float()
        ##
        ## ...但后者只是更简洁，不需要那么多输入。
        ##
        ## 最后，.unsqueeze(1)将torch.arange创建的单列数字转换成一个矩阵，每个索引一行，所有索引
        ## 在同一列。所以如果"max_len"=3，那么我们将得到一个3行1列的矩阵：
        ##
        ## torch.arange(start=0, end=3, step=1, dtype=torch.float).unsqueeze(1)
        ##
        ## ...返回...
        ##
        ## tensor([[0.],
        ##         [1.],
        ##         [2.]])     



        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)

        

        ## 这里开始进行数学计算，确定正弦和余弦曲线上的y轴坐标。
        ##
        ## 《Attention is all you need》中使用的位置编码公式是...
        ##
        ## PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        ## PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        ##
        ## ...在sin()和cos()函数中，我们将"pos"除以一个依赖于索引(i)和所需PE值数量(d_model)的数字。
        ##
        ## 注意：当索引i=0时，我们计算的是**第一对**正弦和余弦曲线的y轴坐标。
        ##       当i=1时，我们计算的是**第二对**正弦和余弦曲线的y轴坐标，依此类推。
        ##
        ## 现在，几乎所有人都首先计算我们用来除"pos"的项，它的代码通常是这样的...
        ##
        ## div_term = torch.exp(torch.arange(start=0, end=d_model, step=2).float() * -(math.log(10000.0) / d_model))
        ##
        ## 至少对我来说，div_term = 1/(10000^(2i/d_model))并不显而易见，原因有几个：
        ##
        ##    1) div_term 将所有内容包装在torch.exp()调用中
        ##    2) 它使用了log()
        ##    2) 各项的顺序不同 
        ##
        ## 这些差异的原因可能是为了防止下溢（即接近0的值）。
        ## 因此，为了展示div_term = 1/(10000^(2i/d_model))，我们可以这样操作...
        ##
        ## 1) 将math.log()替换为torch.log()（这样做需要将10000.0转换为张量，这是我猜测为什么使用math.log()而不是torch.log()的原因）...
        ##
        ## torch.exp(torch.arange(start=0, end=d_model, step=2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        ##
        ## 2) 重排各项...
        ##
        ## torch.exp(-1 * (torch.log(torch.tensor(10000.0)) * torch.arange(start=0, end=d_model, step=2).float() / d_model))
        ##
        ## 3) 用exp(-1 * x) = 1/exp(x)将-1提取出来
        ##
        ## 1/torch.exp(torch.log(torch.tensor(10000.0)) * torch.arange(start=0, end=d_model, step=2).float() / d_model)
        ##
        ## 4) 使用exp(a * b) = exp(a)^b将2i/d_model项提取出来...
        ##
        ## 1/torch.exp(torch.log(torch.tensor(10000.0)))^(torch.arange(start=0, end=d_model, step=2).float() / d_model)
        ##
        ## 5) 使用exp(log(x)) = x恢复分母的原始形式...
        ##
        ## 1/(torch.tensor(10000.0)^(torch.arange(start=0, end=d_model, step=2).float() / d_model))
        ##
        ## 6) 结束。
        ## 
        ## 所以，考虑到这一点，我并不认为下溢问题真的那么严重。事实上，Hugging Face的某个开发者
        ## 也不这么认为，他们在DistilBERT（BERT的简化版本，这是一个Transformer模型）中的位置编码代码
        ## 就是直接按照《Attention is all you need》原文中的公式进行计算的。参考链接：
        ## https://github.com/huggingface/transformers/blob/455c6390938a5c737fa63e78396cedae41e4e87e/src/transformers/modeling_distilbert.py#L53
        ## 因此，我认为我们可以简化代码，不过我还是写了这么多注释来表明这与实际中常见的做法等效。
        ##
        ## 现在，让我们创建一个索引来简化代码...
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        ## 注意：设置step=2将导致与i乘以2时的相同序列号。
        ##       因此我们可以省去一些数学计算，直接将step设为2。

        ## 现在，终于可以创建div_term了...
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        
        ## 现在计算实际的位置信息编码值。记住，'pe'已经初始化为一个全0的矩阵，
        ## 具有max_len（最大输入token数量）行和d_model（每个token的嵌入值数量）列。
        pe[:, 0::2] = torch.sin(position * div_term) ## 从第1列开始的每隔一列，填充sin()值
        pe[:, 1::2] = torch.cos(position * div_term) ## 从第2列开始的每隔一列，填充cos()值
        ## 注意：如果索引'pe[]'看起来有点难懂，接着往下看...
        ##
        ## 首先，我们看一下通用的索引表示法：
        ##
        ## 对于矩阵中的每一行或列，我们可以使用以下索引来选择元素...
        ##
        ## i:j:k = 选择从i到j之间的元素，步长为k。
        ##
        ## ...其中...
        ##
        ## i默认为0
        ## j默认为行列的元素数量
        ## k默认为1
        ##
        ## 现在我们来分析一下具体例子，以帮助理解。
        ##
        ## 我们从：pe[:, 0::2]开始
        ##
        ## 在逗号前面的部分（在这里是':'）指的是我们想选择的行。
        ## 这里的':'意味着"选择所有行"，因为我们没有提供i、j和k的具体值，而是使用默认值。
        ##
        ## 在逗号后面的部分指的是我们想选择的列。
        ## 在这里，我们有'0::2'，意味着从第1列（列=0）开始，直到最后，步长为2，也就是说跳过每一列。
        ##
        ## 现在理解pe[:, 1::2]
        ##
        ## 同样，逗号前的部分指的是行，依然使用默认值选择所有行。
        ## 在逗号后面的部分指的是列。
        ## 这里我们从第2列（列=1）开始，直到最后，步长为2，跳过每一列。
        ##
        ## 注意：使用这种基于':'的表示法叫做“索引”或“切片”。
        
        ## 现在我们"注册'pe'。
        self.register_buffer('pe', pe) ## "register_buffer()"确保'pe'会随模型一起移动
                                       ## 所以如果模型移动到GPU，那么
                                       ## 即使我们不需要优化'pe'，它也会被移动到GPU。
                                       ## 这样，访问'pe'将比让GPU从CPU获取数据更快。

    ## 由于这个类PositionEncoding继承自nn.Module，调用forward()方法
    ## 是默认的行为。
    ## 换句话说，在我们创建一个PositionEncoding()对象后，pe = PositionEncoding()，
    ## 然后pe(word_embeddings)将调用forward()，因此我们将在这里
    ## 将位置编码值加到词嵌入值上
    def forward(self, word_embeddings):
    
        return word_embeddings + self.pe[:word_embeddings.size(0), :] ## word_embeddings.size(0) = 嵌入的数量
                                                                      ## 注意：第二个':'是可选的，我们
                                                                      ## 可以将其重写为：
                                                                      ## self.pe[:word_embeddings.size(0)]

```