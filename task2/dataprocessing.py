from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class Data_init():
    '''随机初始化'''
    def __init__(self,train_data,trained_dict=None,trained_size=-1,case_sent=False):
        '''
        Parameters:
            train_data - list of sentence
            trained_dict - 预训练好的词向量 a dict like [word]:word_vector
            case_sent - 是否区分大小写，默认不区分
        '''
        self.data = train_data
        self.if_case_sent = case_sent 
        self.vocab = dict() #[word]:id
        self.vocab_len = 1 #默认预留 [padding]:0
        self.longest = 0 #最长句子包含的单词数
        self.trained_dict =trained_dict
        self.trained_size = trained_size
        self.embed_mat = []

        self.build_vocab()

    def build_vocab(self):
        '''根据训练数据创建词表'''
        self.embed_mat.append([0] * self.trained_size)  # 先加padding的词向量
        for sentId in range(len(self.data)):
            sent = self.data[sentId]
            if not self.if_case_sent:
                sent=sent.lower() #全部转化为小写
                self.data[sentId] = sent #同时修改原数据
            words = sent.strip().split(" ")
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_len
                    self.vocab_len +=1
                    if self.trained_dict != None: #使用预训练数据
                        if word in self.trained_dict: # 如果有训练好的词向量
                            self.embed_mat.append(self.trained_dict[word])
                        else: # 如果没有 初始化为全0
                            self.embed_mat.append([0]*self.trained_size)
        
    def get_id_seg(self,data):
        '''
        将数据集的句子，根据词表转换成单词ID序列，叠成一个list
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
            return self.embed_mat


class ClsDataset(Dataset):
    '''自定义数据集'''
    def __init__(self,sentence,emotion):
        self.sentence = sentence #句子
        self.emotion = emotion #情感类别
    
    def __getitem__(self,item):
        return self.sentence[item],self.emotion[item]

    def __len__(self):
        return len(self.emotion)

def collate_fn(batch_data):
    '''自定义数据集的内数据返回方式'''
    sentence,emotion = zip(*batch_data)
    sentences = [torch.LongTensor(sent) for sent in sentence] #修改数据类型
    #用0填充padding数据，维度和sentences中最长的数据保持一致
    padded_sents = pad_sequence(sentences,batch_first=True,padding_value=0)
    return torch.LongTensor(padded_sents), torch.LongTensor(emotion)

def dataloader(id_seq,label,batch_size=10):
    '''
    返回自定义数据集的dataloader
        Parameters:
            id_seq - list of word's id in sentences
            label - class of sentences
            batch_size
        return:
            DataLoader
    '''
    dataset = ClsDataset(id_seq,label)
    #  shuffle:每个epoch都随机打乱数据排列再分batch，设置成false，防止之前的排序被打乱
    #  drop_last:不利用最后一个不完整的batch（数据大小不能被batch_size整除）
    return DataLoader(dataset,batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn)