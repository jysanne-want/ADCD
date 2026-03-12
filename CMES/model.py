import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n,topk):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.topk=topk
        self.device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        actual_heads = 3 if self.knowledge_dim % 3 == 0 else 1
        self.multi_attention = MultiHeadAttention(self.knowledge_dim, actual_heads)

        self.predict=PredModel(student_n, exer_n, knowledge_n)
        
    def sinusoidal_positional_encoding(self,inputs):
        seq_len = inputs.shape[1]
        d_model = inputs.shape[2]

        position_enc = torch.zeros((seq_len, d_model))
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        position_enc[:, 0::2] = torch.sin(position * div_term)
        position_enc[:, 1::2] = torch.cos(position * div_term)

        return inputs + position_enc.unsqueeze(0)

    def forward(self, stu_id, exer_id, kn_emb,pos_id=None,ground_truth=None):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''

        stu_emb = torch.sigmoid(self.student_emb(stu_id))                                          

        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))                                       

        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id))                               
        
        if k_difficulty.size(1)==self.topk:
            neg_list, neg_score_list = [], []
            bpr_losses=0

            k_pos=torch.sigmoid(self.k_difficulty(pos_id)) 
            e_disc=torch.sigmoid(self.e_discrimination(pos_id))

            pos_neg_diff=torch.cat((k_difficulty,k_pos.unsqueeze(1)),dim=1)
            pos_neg_disc=torch.cat((e_discrimination,e_disc.unsqueeze(1)),dim=1)

            k_diff_fusion=self.multi_attention(pos_neg_diff).to(self.device)

            for i in range(self.topk+1):
                each_neg=k_diff_fusion[torch.arange(k_diff_fusion.size(0)), i, :]               # [256,123]
                
                neg_disc=pos_neg_disc[torch.arange(e_discrimination.size(0)), i]

                neg_score,bpr_loss=self.predict(stu_id,pos_id,each_neg,neg_disc,ground_truth)
                neg_score_list.append(neg_score)
                bpr_losses+=bpr_loss

                input_x = neg_disc * (stu_emb - each_neg) 
                input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
                input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
                output = torch.sigmoid(self.prednet_full3(input_x))
                
                neg_list.append(output)

            neg_net = torch.stack(neg_list,dim=1)
            neg_score = torch.stack(neg_score_list,dim=1)
            return neg_net,neg_score,bpr_losses
            
        else:
            input_x = e_discrimination * (stu_emb - k_difficulty)
            input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
            input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
            output = torch.sigmoid(self.prednet_full3(input_x))
            return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data

class PredModel(nn.Module):
    def __init__(self, stu_num, exer_num, knowledge_dim):
        
        self.zeta=0.5   

        self.stu_num=stu_num
        self.exer_num=exer_num
        self.knowledge_dim=knowledge_dim

        self.prednet_len1, self.prednet_len2 = 512, 256
        self.prednet_input_len = self.knowledge_dim

        self.device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
        
        super(PredModel, self).__init__()
        self.stu_emb=nn.Embedding(self.stu_num,self.knowledge_dim)
        self.diff_emb=nn.Embedding(self.exer_num,self.knowledge_dim)
        self.disc_emb=nn.Embedding(self.exer_num,self.knowledge_dim)

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        self.loss_function=nn.NLLLoss()
        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, pos_id,neg_diff,neg_disc,ground_truth): 

        stu = torch.sigmoid(self.stu_emb(stu_id)) 
        #  正样本的难度和区分度
        pos_diff=torch.sigmoid(self.diff_emb(pos_id))
        pos_disc=torch.sigmoid(self.disc_emb(pos_id))

        #  计算正样本得分
        pred_pos_score_input=pos_disc * (stu - pos_diff)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(pred_pos_score_input)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        pos_score = torch.sigmoid(self.prednet_full3(input_x))

        #  计算正样本的point损失
        output_0 = torch.ones(pos_score.size()).to(self.device) - pos_score
        output_pos = torch.cat((output_0, pos_score), 1)
        point_loss=self.loss_function(torch.log(output_pos+1e-10), ground_truth)

        #  计算负样本得分
        pred_neg_score_input=neg_disc * (stu - neg_diff)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(pred_neg_score_input)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        neg_score = torch.sigmoid(self.prednet_full3(input_x))

        pos_minus_neg = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))  
        neg_minus_pos = -torch.mean(torch.log(torch.sigmoid(neg_score - pos_score)))  

        bpr_loss=torch.where(ground_truth==1,pos_minus_neg,neg_minus_pos)

        loss=self.zeta*point_loss+(1-self.zeta)*torch.mean(bpr_loss)

        return neg_score.long(),loss

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "输入维度无法被num_heads整除"

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, num, dim = x.size()

        assert dim % self.num_heads == 0, "输入维度无法被num_heads整除"
        head_dim = dim // self.num_heads

        query = self.query(x).reshape(batch_size, num, self.num_heads, head_dim).transpose(1, 2)
        key = self.key(x).reshape(batch_size, num, self.num_heads, head_dim).transpose(1, 2)
        value = self.value(x).reshape(batch_size, num, self.num_heads, head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)

        attended_values = attended_values.transpose(1, 2).contiguous().reshape(batch_size, num, dim)
        output = self.fc(attended_values)

        return output