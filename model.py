#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:23:59 2017

@author: wenyue
"""
from __future__ import unicode_literals, print_function, division



import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


'The following class is encoder without self attention'
class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = n_layers, bidirectional = True)
    
    def display(self):
        for param in self.parameters():
            #print(param)
            print(param.data.size())    
    
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        #print(embedded.data.size())
        for i in range(1):

            output, hidden = self.gru(output, hidden)
            #print(output.data.size())
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(2*self.n_layers, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


'The following class is self attentive encoder'
class selfEncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional = True)
        
        self.r = 7
        self.da = 250
        
        self.S1 = nn.Linear( hidden_size * 2, self.da , bias = False)
        self.S2 = nn.Linear( self.da, self.r, bias = False)
        self.MLP = nn.Linear( self.r * self.hidden_size * 2 , hidden_size*2)
        # da is a hyper parameter
        # r is also a hyper parameter number of hops 
        # 在github那里用了n=30 是因为他的input是一个文章，这里用5 差不多了
        
        # gru 的输出进入到 S1
        # S1 的输出进入S2 加入softmax后用于构建A matrix

        
    def display(self):
        for param in self.parameters():
            #print(param)
            print(param.data.size())    
    
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        #print(input.size( 0 ))
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        output = output.squeeze(1)
        #print('output size')
        #print(output.data.size())
        if use_cuda:
            BM = Variable(torch.zeros(1, self.r * \
                                      self.hidden_size * 2).cuda())
            penal = Variable( torch.zeros( 1 ).cuda() )
            I = Variable( torch.eye( self.r ).cuda() )
        else:
            BM = Variable(torch.zeros(input.size( 0 ), self.r * \
                                      self.hidden_size * 2))
            penal = Variable( torch.zeros( 1 ))
            I = Variable( torch.eye( self.r ))
        weights = {}
        
        # Self attention block
        #print(output.data.size())  # 1,1,600
        s1 = F.tanh(self.S1(output))
        # because it is a linea model, so we hide the WS1 saved in the weights here
        s2 = self.S2(s1).squeeze(1)
        # Ws2 saved here.
        #print('s2 size')
        #print(s2.data.size())
        
        # Attention Weights and Embedding
        A = F.softmax( s2.t() )
        M = torch.mm( A, output )
        BM = M.view(-1)
        
        # Penalization term
        AAT = torch.mm(A, A.t() )
        P = torch.norm( AAT - I, 2)
        penal = P * P
        #weights[1] = A
        
        # MLP BLOC
        output_attn = F.relu(self.MLP( BM ) )
        #output_attn = output_attn.squeeze(0)
        #print(output_attn.data.size())
        return output_attn, hidden, penal#, weights

'Following Decoder is general decoder without attention mechanism'
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional = True)
        self.out = nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



'Following encoder is global attention decoder'     
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 3, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers = n_layers, bidirectional = True)
        self.out = nn.Linear(self.hidden_size*2, self.output_size)
        
    def display(self):
        for param in self.parameters():
            #print(param)
            print(param.data.size())
            
            
    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0], hidden[1]), 1)))
        #print(attn_weights.data)
        #print(encoder_outputs.data)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),  # make it from 1*10 to 1*1*10
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(1):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(2*self.n_layers, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result              