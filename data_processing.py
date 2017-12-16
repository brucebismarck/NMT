#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:14:16 2017

@author: wenyue
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np

import sys
sys.path.append('/home/wenyue/Desktop/data_playground/question_answer')
sys.path

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()  # 这一句很牛逼啊

SOS_token = 0
EOS_token = 1


#This class is used to introduce data

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS":0,"EOS":1}
        self.word2count = {"SOS":1,"EOS":1}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
            
#This function convert unicode to ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )    
    
# Simplify data, Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

#Simplyfy data, remove data with sentence length larger than


# Warning: Something changed here

def filterPair(p, maxlen, minlen, eng_prefixes):
    MAX_LENGTH = maxlen
    MIN_LENGTH = minlen
    'Take pair with words less than MAX_LENGTH'
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        len(p[0].split(' ')) >= MIN_LENGTH and \
        len(p[1].split(' ')) >= MIN_LENGTH #and \
        #p[1].startswith(eng_prefixes)

def filterPairs(pairs, maxlen, minlen, eng_prefixes):
    return [pair for pair in pairs if filterPair(pair, maxlen, minlen, eng_prefixes)]


#Read language file Eng to Fra
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, maxlen, minlen, eng_prefixes, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, maxlen, minlen, eng_prefixes)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



