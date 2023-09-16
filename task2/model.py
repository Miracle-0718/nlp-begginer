import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN_1(nn.Module):
    def __init__(self, vocab_size, embed_size, num_class=5, num_filters=100 ,weight=None, dropout_rate=0.3):
        '''
        Parameters:
            vocab_size - 词表大小（单词数量）
            embed_size - 嵌入词向量维度
            num_class - 标签类别数量
            num_filters - 卷积输出通道数量
            weight - 初始化方式 默认为随机初始化
        '''
        super(CNN_1,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        if weight == None:
            x = nn.init.xavier_normal_(torch.Tensor(vocab_size, embed_size))
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, _weight=x)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, _weight=weight)
        self.dropout = nn.Dropout(dropout_rate) # Dropout层

        self.conv1 = nn.Sequential(
            # Conv2d参数：（输入通道数：1，输出通道数：l_l，卷积核大小：（行数，列数））
            # 对于一句话X只含有一个单词的x:(1*50) 用(2*50)的卷积核无法进行卷积操作 需要用padding扩展行数
            # 例如：X (1*50)->padding=(1，0)变成 3*50 或 padding=(2，0)变成 5*50
            nn.Conv2d(1, num_filters, (2, embed_size), padding=(1, 0)), 
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, num_filters, (3, embed_size), padding=(1, 0)),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, num_filters, (4, embed_size), padding=(2, 0)), 
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, num_filters, (5, embed_size), padding=(2, 0)), 
            nn.ReLU())
        # 全连接层
        self.fc = nn.Linear(4 * num_filters, num_class)
        # softmax
        self.act = nn.Softmax(dim=1)

    def forward(self,x):
        """x:[batch_size,num_words(本句话包含单词数量)]"""
        x = torch.LongTensor(x) #转换为张量类型
        x = self.embedding(x) # 词嵌入
        #x:[batch_size，1,num_words,d]
        x = x.view(x.shape[0], 1, x.shape[1], self.embed_size)  
        x=self.dropout(x)  # dropout层
        #x:[batch_size,num_filters,num_words+2-1(padding作用),1]
        #squeeze挤掉第三维度 x:[batch_size，num_filters，num_words+2-1]
        conv1 = self.conv1(x).squeeze(3)
        #x:[batch_size，num_filters，num_words+2-2]
        conv2 = self.conv2(x).squeeze(3)
        #x:[batch_size，num_filters，num_words+4-3]
        conv3 = self.conv3(x).squeeze(3)
        #x:[batch_size，num_filters，num_words+4-4]
        conv4 = self.conv4(x).squeeze(3)

        #分别对卷积结果的第2维进行pooling 得到4个[batch_size,，num_filters，1]的向量
        pool1 = F.max_pool1d(conv1, conv1.shape[2])
        pool2 = F.max_pool1d(conv2, conv2.shape[2])
        pool3 = F.max_pool1d(conv3, conv3.shape[2])
        pool4 = F.max_pool1d(conv4, conv4.shape[2])

        #cat:[batch_size,num_filters*4,1] squeeze:[batch_size,num_filters*4]
        pool = torch.cat([pool1, pool2, pool3, pool4], 1).squeeze(2)

        # [batch_size, 5]
        out = self.fc(pool)  # 全连接层
        out = self.act(out)
        return out

class CNN_2(nn.Module):
    def __init__(self, vocab_size, embed_size, num_class=5, num_filters=100 ,weight=None, dropout_rate=0.3,
                 kernel_size=[2,3,4]):
        '''
        Parameters:
            vocab_size - 词表大小（单词数量）
            embed_size - 嵌入词向量维度
            num_class - 标签类别数量
            num_filters - 卷积输出通道数量
            weight - 初始化方式 默认为随机初始化
        '''
        super(CNN_2,self).__init__()
        if weight == None:
            x = nn.init.xavier_normal_(torch.Tensor(vocab_size, embed_size))
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, _weight=x)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, _weight=weight)
        self.convs = nn.ModuleList([
            nn.Conv2d(1,num_filters,(k,embed_size),padding=(k-1,0))
            for k in kernel_size
        ])
        self.fc = nn.Linear(len(kernel_size) * num_filters,num_class)
        self.dropout = nn.Dropout(dropout_rate)#dropout层放在前后都可以

    def conv_and_pool(self,x,conv):
        x = F.relu(conv(x).squeeze(3))
        x_max = F.max_pool1d(x,x.size(2)).squeeze(2)
        return x_max
    
    def forward(self,x):
        """x:[batch_size,num_words]"""
        x = torch.LongTensor(x) #转换为张量类型
        #x:[batch_size，1,num_words,d]
        x_embed = self.embedding(x).unsqueeze(1)
        conv_results = [self.conv_and_pool(x_embed,conv) for conv in self.convs]
        out = torch.cat(conv_results,1)
        return self.fc(self.dropout(out))
        


class RNN(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers=1,num_class=5,weight=None,
                 nonlinearity='tanh',batch_first=True,dropout_rate=0):
        '''
        Parameters:
            vocab_size - 词表大小（单词数量）
            embed_size - 嵌入词向量维度
            hidden_size - 隐藏层大小[h,h]
            num_layers - 隐藏层数量，默认一层
            num_class - 标签类别数量
            weight - 词向量初始化方式 默认为随机初始化
            batch_first - 输入数据第一维度是否为batch_size
        '''
        super(RNN,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        if weight == None:
            x = nn.init.xavier_normal_(torch.Tensor(vocab_size, embed_size))
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, _weight=x)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, _weight=weight)
        self.dropout = nn.Dropout(dropout_rate) #dropout
        self.rnn = nn.RNN(input_size=embed_size,hidden_size=hidden_size,num_layers=num_layers,nonlinearity=nonlinearity,
                          batch_first=batch_first,dropout=dropout_rate)
        # 全连接层
        self.fc = nn.Linear(hidden_size,num_class)

    def forward(self,x):
        """x:[batch_size,num_words]"""
        x = torch.LongTensor(x)
        #x:[batch_size,num_words,d]
        x = self.embedding(x)
        x = self.dropout(x)
        #output:[batch_size,num_words,l_h] ht:[1,batch_size,l_h]
        output , hn =self.rnn(x)
        #out[1,batch_size,5] ->[batch_size,5]
        out = self.fc(hn).squeeze(0)
        return out

class LSTM(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers=1,num_class=5,weight=None,
                 bidirectional=False,batch_first=True,dropout_rate=0):
        '''
        Parameters:
            vocab_size - 词表大小（单词数量）
            embed_size - 嵌入词向量维度
            hidden_size - 隐藏层大小[h,h]
            num_layers - 隐藏层数量，默认一层
            num_class - 标签类别数量
            bidirectional - 是否双向
            weight - 词向量初始化方式 默认为随机初始化
        '''
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if weight == None:
            x = nn.init.xavier_normal_(torch.Tensor(vocab_size, embed_size))
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, _weight=x)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, _weight=weight)
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,
                          batch_first=batch_first,dropout=dropout_rate)
        # 全连接层
        if not bidirectional:
            self.fc = nn.Linear(hidden_size,num_class)
        else:
            self.fc = nn.Linear(hidden_size * 2,num_class)
        self.dropout = nn.Dropout(dropout_rate) #dropout
        self.init()

    def init(self):
        '''初始化隐藏层'''
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std,std)

        
    def forward(self,x):
        """x:[batch_size,num_words]"""
        #x:[batch_size,num_words,d]
        x_embed = self.embedding(torch.LongTensor(x))
        #output:[batch_size,num_words,l_h] hn,cn:[1,batch_size,l_h] 
        output,(hn,cn) = self.lstm(x_embed)       
        out = hn.squeeze(0)
        out = self.fc(self.dropout(out))
        return out


