import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class SelfAttentiveLayer(nn.Module):
    def __init__(self, atten_in_dim, attention_unit, num_heads=5):
        super(SelfAttentiveLayer, self).__init__()
        self.ws1 = nn.Linear(atten_in_dim, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, num_heads, bias=False)
        self.tanh = nn.Tanh()
        self.init_weights()
        self.attention_hops = num_heads
        self.dropout = nn.Dropout(p=0.5)

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, x):
        # [batch_sz, seq_len, hid_dim]
        # 200,100,128
        H = x
        A = self.tanh(self.ws1(x))  # 200*100*128 * 128*128 = 200*100*128
        A = self.ws2(A)  # 200*100*128 * 128*5 = 200*100*5
        A = self.softmax(A, 1)  # column-wise softmax
        A = A.transpose(1, 2)  # 200,5,100
        output = torch.bmm(A, H)  # 200*(5*100 * 100*128)=200,5,128
        output = self.dropout(output)

        return output, A

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)


class Time_async(nn.Module):
    def __init__(self, bert_dim=768, atten_in_dim=64, attention_unit=64, num_heads=5):
        super(Time_async, self).__init__()
        self.layer_bert = nn.Linear(bert_dim, atten_in_dim)
        self.bert_atten = SelfAttentiveLayer(atten_in_dim, attention_unit, num_heads)

    def forward(self, x):
        size = x.size()  # 200,768*(context*2+1) (note: each item in batch corresponds to a time step i.e. the utterance in centre of (context*2+1) utterances)
        n_utterances = int(size[-1] / 768)
        bert = []
        for i in range(n_utterances):
            bert.append(F.relu(self.layer_bert(x[:, 768 * i:768 * (i + 1)])).unsqueeze(1))  # 200,768 --> 200,1,64 (note: "shared fully-connected (FC) layer is used to reduce the dimension of each input BERT embedding" --> 1. first reduce dim of each of the 7 sentence embeddings i.e. reduce dims of columns 0-768 (first utterance), then of 769-1537 (second utterance), etc. because the utterances are concatenated along the column dim)
            bert_all = torch.cat((bert), 1)  # 200,(context*2+1),64 (note: (context*2+1) sentence embeddings at each time step (item in batch) with reduced dim 768 --> 64; this will be the input to the self-attentive layer)
        out, A = self.bert_atten(bert_all)  # out: 200, 5, 64  A: 200, 5, (context*2+1)
        return out.view(out.size(0), -1).contiguous(), A  # out: 200, 320  A: 200, 5, (context*2+1)


class Fusion(nn.Module):  # (note: this class defines the whole model)
    def __init__(self, roberta, speechBert, fuse_dim=128, dropout_rate=0.0, output_dim=4, freeze_models=False):  # NOTE: changed output_dim from 5 to 4
        super(Fusion, self).__init__()

        #self.b1 = roberta.eval()
        #self.b2 = speechBert.eval()
        self.b1 = roberta
        self.b2 = speechBert
        self.b3 = Time_async()
        self.layer_cat = nn.Linear(1024 + 768 + 64 * 5, fuse_dim)
        self.layer_out = nn.Linear(fuse_dim, output_dim, bias=False)
        #self.dropout = nn.Dropout(p=dropout_rate)

        #self.layer_cat_out = nn.Linear(1024 + 768 + 64 * 5, output_dim)

        # Freeze models
        if freeze_models:
            for param in roberta.parameters():
                param.requires_grad = False
            for param in speechBert.parameters():
                param.requires_grad = False

    def forward(self, x, y, z):
        x = self.b1.extract_features(x)[:, 0, :]  # x: 200,1024
        y = self.b2.extract_features(y)[:, 0, :]  # y: 200,768
        z, A3 = self.b3(z)  # z: 200,320
        out = torch.cat((x, y, z), 1)  # 200,2112

        out = F.relu(self.layer_cat(out))  # 200,128 # TODO: check if dim fuse_dim=128 is appropriate
        out = self.layer_out(out)  # 200,4

        return out, A3

    # def forward(self, x, y, z):
    #     x = self [:, 0, :]  # x: 200,1024
    #     y = self.b2.extract_features(y)[:, 0, :]  # y: 200,768
    #     z, A3 = self.b3(z)  # z: 200,320
    #     out = torch.cat((x, y, z), 1)  # 200,2112
    #
    #     out = F.relu(self.layer_cat(out))  # 200,128 # TODO: check if dim fuse_dim=128 is appropriate
    #     #out = self.dropout(out) # add dropout
    #     out = F.softmax(self.layer_out(out), dim=-1)  # 200,4 ; (Note: if large margin softmax loss dont apply softmax here)
    #
    #     #out = self.layer_cat_out(out) # instead intermediate linear layer and output layer
    #     return out, A3


class Roberta_RHS(nn.Module):  # (note: this class defines the whole model)
    def __init__(self, roberta, speechBert, fuse_dim=128, dropout_rate=0.0, output_dim=4, freeze_models=False):  # NOTE: changed output_dim from 5 to 4
        super(Roberta_RHS, self).__init__()

        #self.b1 = roberta.eval()
        #self.b2 = speechBert.eval()
        self.b1 = roberta
        self.b2 = speechBert
        self.b3 = Time_async()
        self.layer_cat = nn.Linear(1024 + 64 * 5, fuse_dim)
        self.layer_out = nn.Linear(fuse_dim, output_dim, bias=False)
        #self.dropout = nn.Dropout(p=dropout_rate)

        #self.layer_cat_out = nn.Linear(1024 + 768 + 64 * 5, output_dim)

        # Freeze models
        if freeze_models:
            for param in roberta.parameters():
                param.requires_grad = False
            for param in speechBert.parameters():
                param.requires_grad = False

    def forward(self, x, y, z):
        x = self.b1.extract_features(x)[:, 0, :]  # x: 200,1024
        z, A3 = self.b3(z)  # z: 200,320
        out = torch.cat((x, z), 1)  # 200,2112

        out = F.relu(self.layer_cat(out))  # 200,128 # TODO: check if dim fuse_dim=128 is appropriate
        out = self.layer_out(out)  # 200,4

        return out, A3






# Newbob scheduler
class newbob():
    def __init__(self, optimizer, factor, model):
        self.optimizer = optimizer
        self.factor = factor
        self.model = model
        self.Flast = 0.0
        self.model_epoch = 0
        self.ramp = 0
        self.N = 0
        self.Nmax = 20
        self.threshold = 0.001
        self.model_state = None
        self.optimizer_state = None
        # self.lr = get_lr(optimizer)
        self.lr = optimizer.param_groups[0]['lr']
        self.initlr = 0

    def step(self, F):
        self.N += 1
        print("N:", self.N)
        dF = F - self.Flast
        print(F, self.Flast, dF)
        if dF <= 0 and self.N > 1:
            print("loading model state from epoch:", self.model_epoch)
            self.model.load_state_dict(self.model_state)
            self.optimizer.load_state_dict(self.optimizer_state)
        else:
            print("saving model state")
            self.Flast = F
            self.model_epoch = self.N
            self.model_state = self.model.state_dict()
            self.optimizer_state = self.optimizer.state_dict()
        print("ramp:", self.ramp)

        if self.ramp:
            self.lr = self.lr / 2
        elif dF < self.threshold:
            self.lr = self.lr / 2
            if self.N >= self.Nmax:
                self.ramp = True

        print("lr:", self.lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
