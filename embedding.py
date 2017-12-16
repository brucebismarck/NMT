#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:06:22 2017

@author: wenyue
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class Vocabulary(object):
    def __init__(self, stoi):
        self.embeddings = None
        #self.itos = dict(zip(stoi,range(len(stoi))))
        self.stoi = stoi.word2index
    
    def set_word_embedding(self,embedding_file):
        #'/home/wenyue/Desktop/data_playground//glove.840B.300d.txt'
        f = open(embedding_file,'r', encoding="utf8")
        model = {}
        #count = 0
        for line in f:
            splitLine = re.split(', | |\n',line)
            word = splitLine[0]
            embedding = np.array([np.float32(val) for val in splitLine[1:-1]])
            model[word] = embedding
            
        print(len(self.stoi))
        self.embeddings = np.array([])
        for item in self.stoi:
            if item in model.keys():
                self.embeddings = np.append(self.embeddings, [[model[item]]])
            else:
                self.embeddings = np.append(self.embeddings, \
                                [[np.array(np.random.uniform(-1,1,300), dtype = np.float32)]])
                
                
class Vocabulary_fra(object):
    def __init__(self, stoi):
        self.embeddings = None
        #self.itos = dict(zip(stoi,range(len(stoi))))
        self.stoi = stoi.word2index
    
    def set_word_embedding(self,embedding_file):
        #'/home/wenyue/Desktop/data_playground//glove.840B.300d.txt'
        model = {}
        with open(embedding_file,'r',  encoding="utf8") as myfile:
            for i,line in enumerate(myfile):
                if i != 0:
                    splitLine = re.split(', | |\n',line)
                    word = splitLine[0]
                    embedding = np.array([np.float32(val) for val in splitLine[1:-2]])
                    model[word] = embedding
    
        print(len(self.stoi))
        self.embeddings = np.array([])
        for item in self.stoi:
            if item in model.keys():
                self.embeddings = np.append(self.embeddings, [[model[item]]])
            else:
                self.embeddings = np.append(self.embeddings, \
                                [[np.array(np.random.uniform(-1,1,300), dtype = np.float32)]])