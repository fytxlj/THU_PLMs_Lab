from imp import init_builtin
from re import S
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder"""

    # 调用的时候
    # model = RNNModel("LSTM", vocab_size,embedding_size,embedding_size,2,dropout=0.5)
    def __init__(self,
                 rnn_type,
                 token_num,
                 input_dimension,
                 hidden_num,
                 layer_num,
                 dropout=0.5,
                 tie_weight=False):
        super(RNNModel, self).__init__()
        self.token_num = token_num
        # nn.Embedding是一个简单的存储固定大小的词典的嵌入向量的查找表。
        # 给一个编号，嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系。
        # nn.Embedding(num_embedding词典长度，embedding_dim向量维度)
        self.encoder = nn.Embedding(token_num, input_dimension)
        # 将长度从hidden_num转换成token_num
        self.decoder = nn.Linear(hidden_num, token_num)
        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.rnn_type = rnn_type
        self.hidden_num = hidden_num
        self.layer_num = layer_num
        if rnn_type in ['LSTM', 'GRU']:
            # getattr用于返回一个对象的属性值
            self.rnn = getattr(nn, rnn_type)(input_dimension,
                                             hidden_num,
                                             layer_num,
                                             dropout=dropout)
        else:
            try:
                nbnlinearity = {
                    'RNN_TANH': 'tanh',
                    'RNN_RELU': 'relu'
                }[rnn_type]
            except:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                    options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        # sst2的decoder不应在整个词表上预测，而是预测0/1的概率，因此下面ntoken可能需要改成2，即2分类

    # 初始化参数
    def init_weights(self):

        # nn.init用于参数初始化
        # 均匀分布U(a,b) torch.nn.init.uniform_(tensor, a=0, b=1)
        # 正态分布N(mean,std) torch.nn.init.normal_(tensor, mean=0, std=1)
        # 初始化为常数torch.nn.init.constant_(tensor, val)
        """
        xavier初始化
        以sigmod为例：
        如果初始化值很小，那么随着层数的传递，方差趋近于0 ，接近于线性，失去了非线性特征
        如果初始值很大，那么随着层数的传递，方差会迅速增加，sigmod在大输入值的时候倒数趋近于0 
        ，导致反向传播的时候会遇到梯度消失的问题。
        xavier初始化保证每一层网络的输入和输出的方差相同
        均匀分布 ~ U(−a,a)  torch.nn.init.xavier_uniform_(tensor, gain=1)
        正态分布 ~ N(0,std) torch.nn.init.xavier_normal_(tensor, gain=1)
        """
        """
        kaiming初始化
        xavier在tanh中表现的很好，但是在Relu表现的很差，何凯明提出了针对于Relu的初始化方法。
        在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0，所以，要保持方差不变，只需要在 Xavier 的基础上再除以2
        均匀分布 ~ U(−a,a) torch.nn.init.kaiming_uniform_(tensor, a=0, mode=‘fan_in’, nonlinearity=‘leaky_relu’)
        正态分布 ~ N(0,std) torch.nn.init.kaiming_normal_(tensor, a=0, mode=‘fan_in’, nonlinearity=‘leaky_relu’)
        """
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        # 由于我们做的不是生成任务，hidden在这里面没用了，我们不需要上一个状态的hidden，每个样本都是独立的
        embedding = self.drop(self.encoder(input))
        output, hidden = self.rnn(embedding, hidden)
        # 这里面也不需要hidden作为输入了，输出也不用要hidden
        output = self.drop(embedding)
        # 补充代码，从上面的output中，抽取最后一个词的输出作为最终输出。要注意考虑到序列的真实长度。最后得到一个shape是bsz, nhid的tensor
        # 提示：output = output[real_seq_lens - 1, torch.arange(output.shape[1]), :]
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.token_num)
        return decoded, hidden
        # 不再需要输出hidden；最终输出的shape是bsz, 2

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if (self.rnn_type == 'LSTM'):
            return (weight.new_zeros(self.layer_num, bsz, self.hidden_num),
                    weight.new_zeros(self.layer_num, bsz, self.hidden_num))

        else:
            return weight.new_zeros(self.layer_num, self, self.hidden_num)
