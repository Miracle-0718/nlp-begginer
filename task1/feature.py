import numpy as np

class BagOfWords:
    """Bag of words"""
    def __init__(self,train_data,case_sent=False):
        '''
        Parameters:
            train_data - list of sentence
            case_sent - 是否区分大小写，默认不区分
        '''
        self.data = train_data 
        self.if_case_sent = case_sent 
        self.vocab = dict() #词汇表 [word]:id
        self.vocab_len = 0 #词汇表长度

    def build_vocab(self):
        '''
        构造词典
        '''
        for sentId in range(len(self.data)):
            # 遍历数据集 获取单个句子
            sent = self.data[sentId]
            # 是否统一大小写
            if not self.if_case_sent:
                sent=sent.lower() #全部转化为小写
                self.data[sentId] = sent #同时修改原数据
            # 句子分割为单词列表
            words = sent.strip().split(" ")
            # 构造词典
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_len
                    self.vocab_len +=1

    def get_feature_matrix(self):
        '''
        构造特征矩阵 [data.size,vocab_len]
        矩阵每一行为一个句子，单词出现在句子中，则词表中的对应位置+1
        '''
        self.build_vocab()
        # print(self.vocab)
        feature_mat = np.zeros((len(self.data),self.vocab_len))
        for sentId in range(len(self.data)):
            words = self.data[sentId].strip().split(" ")
            for word in words:
                feature_mat[sentId][self.vocab[word]] += 1
        return feature_mat


class Ngram:
    """n-gram"""
    def __init__(self,train_data,N=2,case_sent=False):
        '''
        Parameters:
            train_data - list of sentence
            case_sent - 是否区分大小写，默认不区分
        N - n of n-gram
        '''
        self.data = train_data 
        self.if_case_sent = case_sent
        self.N = N 
        self.feature_map = dict() #特征表 [feature]:id
        self.map_len = 0 #特征表长度

    def build_feature_map(self):
        for sentId in range(len(self.data)):
            sent = self.data[sentId]
            if not self.if_case_sent:
                sent=sent.lower() #全部转化为小写
                self.data[sentId] = sent #同时修改原数据
            words = sent.strip().split(" ")
            # 滑动N大小的窗口截取短语N-gram
            for gram in range(1,self.N+1):
                for i in range(len(words)-gram+1):
                    feature = '_'.join(words[i:i+gram])
                    if feature not in self.feature_map:
                        self.feature_map[feature] = self.map_len
                        self.map_len+=1

    def get_feature_matrix(self):
        self.build_feature_map()
        # print(self.feature_map)
        feature_mat = np.zeros((len(self.data),self.map_len))
        for sentId in range(len(self.data)):
            words = self.data[sentId].strip().split(" ")
            for gram in range(1,self.N+1):
                for i in range(len(words)-gram+1):
                    feature = '_'.join(words[i:i+gram])
                    feature_mat[sentId][self.feature_map[feature]] +=1
        return feature_mat
