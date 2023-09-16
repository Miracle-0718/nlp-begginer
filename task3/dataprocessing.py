from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

class Data_init():
    '''数据初始化'''
    def __init__(self,train_data,trained_dict=None,dict_size=-1,case_sent=False):
        '''
        Parameters:
            train_data - list of sentence
            trained_dict - 预训练好的词向量 a dict like [word]:word_vector
            case_sent - 是否区分大小写，默认不区分
        '''
        self.data = train_data
        self.if_case_sent = case_sent
        #直接先复制已有的词汇表 
        self.vocab = dict() #[word]:id
        self.vocab_len = 1 #预留 [padding]:0
        self.longest = 0 #最长句子包含的单词数
        self.trained_dict =trained_dict
        self.dict_size = dict_size
        self.embed_mat = []

        self.build_vocab()

    def build_vocab(self):
        '''根据训练数据创建词表'''
        self.embed_mat.append([0] * self.dict_size)  # 添加padding的词向量
        # 获取所有预训练词向量放入词表
        for word,vector in self.trained_dict.items():
            self.vocab[word] = self.vocab_len
            self.vocab_len += 1
            self.embed_mat.append(vector)

        # 增加训练集中存在的未知单词，初始化为全0向量
        for sentId in range(len(self.data)):
            sent = self.data[sentId]
            if not self.if_case_sent:
                sent=sent.lower() #全部转化为小写
                self.data[sentId] = sent #同时修改原数据
            words = sent.strip().split(" ")
            for word in words:
                if word not in self.vocab:
                    # 不在已有词汇表里 初始化为全0 词表加长
                    self.vocab[word] = self.vocab_len
                    self.vocab_len += 1
                    self.embed_mat.append([0]*self.dict_size)
        
    def get_id_seg(self,data):
        '''
        将给定数据集中的句子，根据词表转换成单词ID序列，叠成一个list
        Parameters:
            data - list of sentence
        return:
            id_seq - list of word's id in sentences
        '''
        id_seg = []
        longest = 0
        for sent in data:
            if not self.if_case_sent:
                sent=sent.lower() #全部转化为小写
            words = sent.split(" ")
            seg = [self.vocab[word] for word in words]
            longest = max(longest,len(seg))
            id_seg.append(seg)
        self.longest = longest
        return id_seg  

    def get_vocab_size(self):
        return self.vocab_len

    def get_longest(self):
        return self.longest

    def get_embed_mat(self):
        if self.trained_dict != None: 
            return np.array(self.embed_mat)


class ClsDataset(Dataset):
    '''文本蕴含关系分类数据集'''
    def __init__(self,sentence1,sentence2,relation):
        self.sentence1 = sentence1 #前提
        self.sentence2 = sentence2 #假设
        self.relation = relation #蕴涵类别
    
    def __getitem__(self,item):
        return self.sentence1[item],self.sentence2[item],self.relation[item]

    def __len__(self):
        return len(self.relation)

def collate_fn(batch_data):
    '''自定义数据集的内数据返回方式'''
    sents1,sents2,relation = zip(*batch_data)
    sentences1 = [torch.LongTensor(sent) for sent in sents1] #修改数据类型
    sentences2 = [torch.LongTensor(sent) for sent in sents2]
    #用0填充padding数据，维度和sentences中最长的数据保持一致
    padded_sents1 = pad_sequence(sentences1,batch_first=True,padding_value=0)
    padded_sents2 = pad_sequence(sentences2,batch_first=True,padding_value=0)
    return torch.LongTensor(padded_sents1), torch.LongTensor(padded_sents2), torch.LongTensor(relation)

def dataloader(x1,x2,y,batch_size=10):
    '''
    返回自定义数据集的dataloader
        Parameters:
            id_seq - list of word's id in sentences
            label - class of relations
            batch_size
        return:
            DataLoader
    '''
    dataset = ClsDataset(x1,x2,y)
    #  shuffle:每个epoch都随机打乱数据排列再分batch，设置成false，防止之前的排序被打乱
    #  drop_last:不利用最后一个不完整的batch（数据大小不能被batch_size整除）
    return DataLoader(dataset,batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn)