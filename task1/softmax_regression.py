import numpy as np
import random

def sigmod(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z -= np.max(z,axis=1,keepdims=True) #减去改行最大值
    z = np.exp(z)
    z = z/ np.sum(z,axis=1,keepdims=True)
    return z

class Softmax_regression:
    def __init__(self) :
        self.num_sample = None #数据样本数量
        self.num_features = None #特征数量
        self.num_classes = None #类别数量
        self.batch_size = None
        self.w = None
    
    def get_y_one_hot(self,y):
        y_one_hot = np.zeros((len(y),self.num_classes))
        for i in range(self.batch_size):
            y_one_hot[i][y[i]] = 1
        return y_one_hot

    def fit(self,X, y, batch_size = -1, num_classes=5, learning_rate=0.01, epochs=10):
        '''
        :param X: [num_sample, num_features]
        :param y: [num_sample, 1]
        :param w: [num_classes, num_features]
        :return:
        '''
        if len(X) != len(y):
            raise Exception("Sample size does not match!")  # 样本个数不匹配
        self.num_sample , self.num_features = X.shape
        self.num_classes = num_classes
        if batch_size == -1:
            self.batch_size = self.num_sample #默认为整批量梯度
        else:
            self.batch_size = batch_size
        self.w =  np.random.randn( self.num_classes , self.num_features)

        y_one_hot = self.get_y_one_hot(y)

        loss_history = []

        for t in range(epochs):
            #抽取batch里的样本
            batchIds = random.sample(list(range(self.num_sample)),self.batch_size)
            #前x向计算损失
            loss = 0 #本轮总损失
            probs = X.dot(self.w.T)
            probs = sigmod(probs)
            for i in batchIds:
                loss -= np.log(probs[i][y[i]]) #单个样本损失
            #反向传播
            weight_update = np.zeros_like(self.w)
            for i in batchIds:
                weight_update -= X[i].reshape(1,self.num_features).T.dot((y_one_hot[i] - probs[i]).reshape(1,self.num_classes)).T
            weight_update = weight_update/self.batch_size
            self.w -= weight_update * learning_rate

            loss /= self.batch_size
            loss_history.append(loss)
            if (t+1) % 10 == 0 : #每10轮输出一次loss
                print("epoch {} loss {}".format(t+1, loss))
        
        return loss_history

    def predict(self,X):
        prob = softmax(X.dot(self.w.T))
        return prob.argmax(axis=1)

    def score(self,X,y):
        pred = self.predict(X)
        return np.sum(pred.reshape(y.shape) == y) / y.shape[0]
