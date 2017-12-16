#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:45:40 2017

@author: wenyue
"""
#####################
'Data processing part'
####################

import sys
sys.path.append('/home/wenyue/Desktop/data_playground/question_answer')
sys.path

from data_processing import *

MAX_LENGTH = 20
MIN_LENGTH = 8

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

input_lang, output_lang, pairs = prepareData('eng', 'fra', MAX_LENGTH, MIN_LENGTH, eng_prefixes ,True)

##################
'Embedding part!'
##################
from embedding import *
eng_vocab = Vocabulary(output_lang)             
eng_vocab.set_word_embedding('/home/wenyue/Desktop/data_playground/hw2/glove.6B.300d.txt')             
eng_vocab.embeddings = eng_vocab.embeddings.reshape((eng_vocab.embeddings.shape[0]//300,300))
eng_embeddings = eng_vocab.embeddings

fra_vocab = Vocabulary_fra(input_lang)             
fra_vocab.set_word_embedding('/home/wenyue/Desktop/data_playground/hw2/wiki.fr.vec') 
fra_vocab.embeddings = fra_vocab.embeddings.reshape((fra_vocab.embeddings.shape[0]//300,300))
fra_embeddings = vocab.embeddings


#######################
'model training part!'
#######################

from train import *
from model import *
use_cuda = True
MAX_LENGTH = 20
teacher_forcing_ratio = 0.5  
hidden_size = 300
Source_encoding = False
Target_encoding = False

'Change the encoder and decoder here. May need to do a little bit change in train part because of the input output shape'
encoder1 = EncoderRNN(input_lang.n_words, hidden_size,3)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,3, dropout_p=0.1, max_length=MAX_LENGTH)

if Source_encoding:
    encoder1.embedding.weight.data = torch.from_numpy(fra_embeddings).float()
if Target_encoding:
    attn_decoder1.embedding.weight.data = torch.from_numpy(eng_embeddings).float()

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()


trainIters(encoder1, attn_decoder1, 200000, pairs, input_lang, output_lang, print_every=5000, maxlen = MAX_LENGTH)


##########################
'Evaluation and viz part'
##########################

from train import *
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
import torch
SOS_token = 0
EOS_token = 1
import random

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size*2))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)



evaluateRandomly(encoder1, attn_decoder1)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")










