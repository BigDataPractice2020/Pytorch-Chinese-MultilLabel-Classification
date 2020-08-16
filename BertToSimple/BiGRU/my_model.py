import torch
# coding: utf-8
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from model.modeling_albert import AlbertConfig, AlbertForSequenceClassification
class DistillModel(nn.Module):
    def __init__(self,args,config):
        super(DistillModel, self).__init__()
        raw_model = AlbertForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                           from_tf=bool('.ckpt' in args.model_name_or_path),
                                                            config=config)
        print(raw_model)
        self.albert_embedding = list(raw_model.children())[0]
        self.albert_embedding = list(self.albert_embedding.children())[0]
        self.albert_embedding = list(self.albert_embedding.children())[0]
        print(config)
        self.drop = nn.Dropout(0.1)
        self.gru = nn.GRU(input_size=128, hidden_size=1024,
                          num_layers=1, batch_first=True, bidirectional=True)
        # self.output_layer = nn.Sequential(nn.Linear())
        # 只要Embedding结构
        #print(self.albert_embedding)
        self.output_layer = nn.Sequential(nn.Linear(2*1024,2*1024),
                                          nn.ReLU(True),
                                          nn.Linear(2*1024,21))



    def init_weights(self):
        init_uniform = 0.1
        # baseline initial===========================================
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        # input_idx->[batch_size,64]
        # attention_mask->[batch_size,64]
        len_list  = []
        device = input_ids.device
        for i in range(attention_mask.size(0)):
            L = torch.sum(attention_mask[i]==1).item()
            len_list.append(L)
        embeddings = self.albert_embedding(input_ids)
        # embeddings->[batch_size,64,128]
        len_list = torch.Tensor(len_list).to(device)
        X = pack_padded_sequence(embeddings,len_list,batch_first=True,enforce_sorted=False)
        output_and_len,hid = self.gru(X)
        output,len_list = pad_packed_sequence(output_and_len,batch_first=True)
        # print("t5-t4",t5-t4)
        # output-> [batch_ize,seq_len,2*1024]
        output_list = []
        for i in range(output.size(0)):
            output_list.append(output[i,len_list[i]-1,:].unsqueeze(0))
        output = torch.cat(output_list,0)
        logits = self.output_layer(output)
        if labels != None:
            mse_loss = MSELoss()
            cross_loss = CrossEntropyLoss()
            loss = 0.2*cross_loss(logits,labels[:,0].long())+0.8*mse_loss(logits,labels[:,1:])
            return (loss,logits)
        else:
            return (None,logits)

    def init_hidden(self,batch_size,use_gpu):
        weight = next(self.parameters()).data
        if use_gpu == True:
            hidden_and_C = (weight.new(self.nlayers,batch_size,self.nhid).zero_().cuda(),
                            weight.new(self.nlayers,batch_size,self.nhid).zero_().cuda())
        else:
            hidden_and_C = (weight.new(self.nlayers, batch_size, self.nhid).zero_(),
                            weight.new(self.nlayers, batch_size, self.nhid).zero_())
        return hidden_and_C



class DistillModel1(nn.Module):
    def __init__(self,args,config):
        super(DistillModel1, self).__init__()
        raw_model = AlbertForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                           from_tf=bool('.ckpt' in args.model_name_or_path),
                                                            config=config)
        self.albert_embedding = list(raw_model.children())[0]
        self.albert_embedding = list(self.albert_embedding.children())[0]
        self.albert_embedding = list(self.albert_embedding.children())[0]
        print(config)
        self.drop = nn.Dropout(0.1)
        self.gru = nn.GRU(input_size=128, hidden_size=1024,
                          num_layers=1, batch_first=True, bidirectional=True)
        # self.output_layer = nn.Sequential(nn.Linear())
        # 只要Embedding结构
        #print(self.albert_embedding)
        self.output_layer = nn.Sequential(nn.Linear(2*1024,2*1024),
                                          nn.ReLU(True),
                                          nn.Linear(2*1024,21))



    def init_weights(self):
        init_uniform = 0.1
        # baseline initial===========================================
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        # input_idx->[batch_size,64]
        # attention_mask->[batch_size,64]
        len_list  = []
        device = input_ids.device
        for i in range(attention_mask.size(0)):
            L = torch.sum(attention_mask[i]==1).item()
            len_list.append(L)
        embeddings = self.albert_embedding(input_ids)
        # embeddings->[batch_size,64,128]
        len_list = torch.Tensor(len_list).to(device)
        X = pack_padded_sequence(embeddings,len_list,batch_first=True,enforce_sorted=False)
        output_and_len,hid = self.gru(X)
        output,len_list = pad_packed_sequence(output_and_len,batch_first=True)
        # print("t5-t4",t5-t4)
        # output-> [batch_ize,seq_len,2*1024]
        output = self.output_layer(output)
        # 取有效序列最后一位？
        output_list = []
        for i in range(output.size(0)):
            output_list.append(output[i,len_list[i]-1,:].unsqueeze(0))
        logits = torch.cat(output_list,0).to(device)
        if labels != None:
            mse_loss = MSELoss()
            cross_loss = CrossEntropyLoss()
            loss = 0.2*cross_loss(logits,labels[:,0].long())+0.8*mse_loss(logits,labels[:,1:])
            return (loss,logits)
        else:
            return (None,logits)

    def init_hidden(self,batch_size,use_gpu):
        weight = next(self.parameters()).data
        if use_gpu == True:
            hidden_and_C = (weight.new(self.nlayers,batch_size,self.nhid).zero_().cuda(),
                            weight.new(self.nlayers,batch_size,self.nhid).zero_().cuda())
        else:
            hidden_and_C = (weight.new(self.nlayers, batch_size, self.nhid).zero_(),
                            weight.new(self.nlayers, batch_size, self.nhid).zero_())
        return hidden_and_C