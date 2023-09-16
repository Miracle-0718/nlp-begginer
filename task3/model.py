import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding_Layer(nn.Module):

    def __init__(self, vocab_size, embedding_size, weight=None, drop_out=0.0):
        super(Embedding_Layer,self).__init__()
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(vocab_size,embedding_size)).cuda()
            self.emdedding = nn.Embedding(vocab_size,embedding_size,_weight = x).cuda()
        else:
            self.emdedding = nn.Embedding(vocab_size,embedding_size,_weight = weight).cuda()

        self.dropout = nn.Dropout(drop_out)

    def forward(self,x):
        x = torch.LongTensor(x).cuda()
        x = self.emdedding(x)
        x = self.dropout(x)
        return x
    
class Encoding_Layer(nn.Module):

    def __init__(self,embedding_size, hidden_size, num_layers=1):
        super(Encoding_Layer,self).__init__()
        self.lstm = nn.LSTM(input_size = embedding_size,hidden_size= hidden_size,num_layers=num_layers,
                            bidirectional = True, batch_first = True).cuda()
        
    def forward(self,x):
        x, _ = self.lstm(x)
        # [batch_size, length, hidden_size * 2]
        return x
    

class  LocalInference_Layer(nn.Module):
    def __init__(self):
        super(LocalInference_Layer,self).__init__()
        self.softmax1 = nn.Softmax(dim=1).cuda()
        self.softmax2 = nn.Softmax(dim=2).cuda()

    def forward(self, p, h):
        # [batch_size, length1, length2]
        e = torch.matmul(p, h.transpose(1,2)).cuda()
        # h_score :h中的单个单词在p上的注意力
        # p_score :p中的单个单词在h上的注意力
        h_score, p_score = self.softmax1(e), self.softmax2(e)
        p_ = p_score.bmm(h) # [batch_size,length1, hidden_size*2]
        h_ = h_score.transpose(1,2).bmm(p) # [batch_size,length2,hidden_size*2]

        m_p =torch.cat((p, p_, p - p_, p * p_),dim=2) # [batch_size,length1, hidden_size*8]
        m_h =torch.cat((h, h_, h - h_, h * h_), dim=2) # [batch_size,length2,hidden_size*8]

        return m_p, m_h
    
class Composition_Layer(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,dropout=0.0):
        super(Composition_Layer,self).__init__()
        self.linear = nn.Linear(input_size,output_size).cuda()
        self.lstm = nn.LSTM(output_size,hidden_size,bidirectional = True,batch_first =True).cuda()
        self.dropout =  nn.Dropout(dropout).cuda()

    def forward(self,x):
        x = self.linear(x)
        x = self.dropout(x)
        #  self.lstm.flatten_parameters() #什么用？
        x, _ = self.lstm(x)

        return x # [batch_size,output_size,hidden_size*2]
    
class Pooling_Layer(nn.Module):

    def __init__(self):
        super(Pooling_Layer,self).__init__()

    def forward(self,x):
        v_avg = x.sum(1) / x.shape[1] # [batch_size, hidden_size*2]
        v_max = x.max(1)[0] # [batch_size, hidden_size*2]

        return torch.cat((v_avg, v_max), dim = -1) # [batch_size , hidden_size*4]

    
class InferenceComposition_Layer(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,dropout=0.0):
        super(InferenceComposition_Layer,self).__init__()
        self.composition = Composition_Layer(input_size,output_size,hidden_size,dropout)
        self.pooling = Pooling_Layer()

    def forward(self,m_p,m_h):

        v_p , v_h = self.composition(m_p),self.composition(m_h) # [batch_size,output_size,hidden_size*2]
        v_p , v_h = self.pooling(v_p), self.pooling(v_h) # [batch_size , hidden_size*4]

        return torch.cat((v_p, v_h), dim = 1) # [batch_size , hidden_size*8]
    
class Output_Layer(nn.Module):
     
    def __init__(self, input_size, output_size, type_num, dropout=0.0):
        super(Output_Layer, self).__init__()
 
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, output_size, nn.Tanh),
            nn.Linear(output_size, type_num)
        ).cuda()
 
    def forward(self, x):
        return self.mlp(x)

class ESIM(nn.Module):
     
    def __init__(self,vocab_size,embedding_size,hidden_size,type_num=4,weight=None,dropout=0.0):
        super(ESIM, self).__init__()
 
        self.embed = Embedding_Layer(vocab_size, embedding_size, weight=weight)
        self.encoder = Encoding_Layer(embedding_size, hidden_size)
        self.inference = LocalInference_Layer()
        self.inference_composition = InferenceComposition_Layer(hidden_size * 8, hidden_size, hidden_size, dropout)
        self.out = Output_Layer(hidden_size * 8, hidden_size, type_num, dropout)
 
    def forward(self, p, h):
        # [batch_size, length]
        p_embeded = self.embed(p)
        h_embeded = self.embed(h)
        # [batch_size, length, embedding_size]

        p_ = self.encoder(p_embeded)
        h_ = self.encoder(h_embeded)
        # [batch_size, length, hidden_size *2]
 
        m_p, m_h = self.inference(p_, h_)
        # [batch_size, length, hidden_size *8]
 
        v = self.inference_composition(m_p, m_h)
        # [batch_size, hidden_size *8]

        out = self.out(v) # [batch_size, type_num]
 
        return out
    
    
def main():
    vocab_size = 25
    esim = ESIM(vocab_size=vocab_size,
                embedding_size=10,
                hidden_size=16,
                type_num=4,)

    x1 = torch.randint(vocab_size,[10,3])
    x2 = torch.randint(vocab_size,[10,4])

    print(esim(x1,x2).shape)

if __name__=='__main__':
    main()
