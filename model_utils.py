#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:25:08 2019

@author: Morten

Utilities for setting up the Hierarchical Attention Network 
and extracting information.

Implementation of full model is inspired by github user minqi (https://github.com/minqi)
and his HNATT work.

Attention layer is provided courtesy of Felbo, B., from his work on DeepMoji.


"""
import numpy as np
import json
import pickle

#Keras utility
from keras.engine.topology import Layer
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dense, GRU, Bidirectional, TimeDistributed
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras import initializers as initializers, regularizers, constraints

#DeepMoji attention
from attlayer import AttentionWeightedAverage


LEN_TWEET        = 30
MAX_NUM_TWEETS   = 989
VOCAB_SIZE       = 50000 # Inferred from data preprocessing
EMBED_DIM        = 50 # Decreased to 1/4th for computation time
MOMENTUM         = 0.9 # Hierarchical Attention Networks for Document Classification
OPTIMIZER        = optimizers.SGD # Hierarchical Attention Networks for Document Classification
DENSE_ACTIVATION = 'tanh' # Hierarchical Attention Networks for Document Classification
GRU_UNITS        = 50 # Hierarchical Attention Networks for Document Classification
L2_REG           = regularizers.l2(1e-10) # Arbritarily chosen to force weights down in GRU
BATCH_SIZE       = 64
DENSE_UNITS      = 100
EPOCHS           = 1
LEARNING_RATE    = 0.01 
NUM_LABELS       = 49


def build_full_model(DENSE_UNITS    = DENSE_UNITS, 
                LEARNING_RATE  = LEARNING_RATE,
                ACTIVATION     = DENSE_ACTIVATION,
                VOCAB_SIZE     = VOCAB_SIZE, 
                EMBED_DIM      = EMBED_DIM,
                OPTIMIZER      = OPTIMIZER,
                MOMENTUM       = MOMENTUM,
                LEN_TWEET      = LEN_TWEET,
                MAX_NUM_TWEETS = MAX_NUM_TWEETS,
                GRU_UNITS      = GRU_UNITS,
                L2_REG         = L2_REG,
                NUM_LABELS     = NUM_LABELS,
                save_word_model= False):
    """
    Model architecture for the Hierarchical Attention Network.
    Create a list as
    word_model_containter = [0]
    before calling build_model(save_word_model = True)
    to extract the word_model preceding the tweet level layers.
    (An ugly hack)
    """
    
    #Word layer
    word_input     = Input(shape = (LEN_TWEET, ),
                           name  = "word_input",
                           dtype = "uint16")
    
    word_embedding = Embedding(input_dim = VOCAB_SIZE, 
                               output_dim = EMBED_DIM, 
                               input_length = LEN_TWEET)(word_input)
    
    word_encoding  = Bidirectional(GRU(units = GRU_UNITS, 
                                       input_shape = (MAX_NUM_TWEETS, EMBED_DIM),
                                       return_sequences=True, 
                                       kernel_regularizer=L2_REG))(word_embedding)
    
    word_dense     = TimeDistributed(Dense(DENSE_UNITS , 
                                           activation = ACTIVATION), 
                                     name = 'word_dense')(word_encoding) #Name layer to extract for viz
    
    word_att       = AttentionWeightedAverage(name = 'word_att')(word_dense)   
    word_model     = Model(word_input, word_att)
    
    if save_word_model: #hacks for saving word_model
        print('Saving Word Model')
        word_model_container = [word_model]
    
    #Sentence layer
    tweet_input     = Input(shape=(MAX_NUM_TWEETS, LEN_TWEET), 
                           dtype="int32")
    
    tweet_encoding  = TimeDistributed(word_model)(tweet_input)
    
    tweet_lstm      = Bidirectional(GRU(units = GRU_UNITS, 
                                       return_sequences=True, 
                                       kernel_regularizer= L2_REG))(tweet_encoding)
    
    tweet_dense     = TimeDistributed(Dense(DENSE_UNITS , 
                                           activation = ACTIVATION), 
                                     name = 'tweet_dense')(tweet_lstm)
    
    tweet_att       = AttentionWeightedAverage(name = 'tweet_att')(tweet_dense)
    preds           = Dense(NUM_LABELS, activation='softmax')(tweet_att)
    model           = Model(tweet_input, preds)
    
    #Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer = OPTIMIZER(momentum = MOMENTUM, lr = LEARNING_RATE), 
                  metrics=['acc'])
    
    return model

def build_word_model(DENSE_UNITS    = DENSE_UNITS, 
                     LEARNING_RATE  = LEARNING_RATE,
                     ACTIVATION     = DENSE_ACTIVATION,
                     VOCAB_SIZE     = VOCAB_SIZE, 
                     EMBED_DIM      = EMBED_DIM,
                     OPTIMIZER      = OPTIMIZER,
                     MOMENTUM       = MOMENTUM,
                     LEN_TWEET      = LEN_TWEET,
                     MAX_NUM_TWEETS = MAX_NUM_TWEETS,
                     GRU_UNITS      = GRU_UNITS,
                     L2_REG         = L2_REG,
                     NUM_LABELS     = NUM_LABELS):
    """
    Build the word_model where weights can be loaded afterwards.
    Allows for word level attention visualization.
    """
    
    #Word layer
    word_input     = Input(shape = (LEN_TWEET, ),
                           name  = "word_input",
                           dtype = "uint16")
    
    word_embedding = Embedding(input_dim = VOCAB_SIZE, 
                               output_dim = EMBED_DIM, 
                               input_length = LEN_TWEET)(word_input)
    
    word_encoding  = Bidirectional(GRU(units = GRU_UNITS, 
                                       input_shape = (MAX_NUM_TWEETS, EMBED_DIM),
                                       return_sequences=True, 
                                       kernel_regularizer=L2_REG))(word_embedding)
    
    word_dense     = TimeDistributed(Dense(DENSE_UNITS , 
                                           activation = ACTIVATION), 
                                     name = 'word_dense')(word_encoding) #Name layer to extract for viz
    
    word_att       = AttentionWeightedAverage(name = 'word_att')(word_dense)   
    word_model     = Model(word_input, word_att)
    
    return word_model
    
    #Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer = OPTIMIZER(momentum = MOMENTUM, lr = LEARNING_RATE), 
                  metrics=['acc'])
    
    return word_model



def attention_map_word(tokenized_text, word_model):    
    """
    Returns attention weights on a word level for the Hierarchical Attention Network
    """
    
    word_dense    = Model(inputs = word_model.input,
                          outputs = word_model.get_layer('word_dense').output)
    
    u_weights  = word_model.get_layer('word_att').get_weights()[0]
    u_text = word_dense.predict(np.reshape(tokenized_text, (1,30)))
    
    att_weights = np.exp(np.dot(u_text, u_weights)) / np.sum(np.exp(np.dot(u_text, u_weights)), axis = 1)
    
    return np.squeeze(att_weights)

def attention_map_tweet(tweets, model):
    """
    Returns attention weights on a tweet level for the Hierarchical Attention Network for one user
    """
    
    tweet_dense = Model(inputs = model.input,
                            outputs = model.get_layer('tweet_dense').output)
    
    u_weights   = model.get_layer('tweet_att').get_weights()[0]
    u_tweets    = tweet_dense.predict(np.reshape(tweets, (1, 989, 30)))
    
    att_weights = np.exp(np.dot(u_tweets, u_weights)) / np.sum(np.exp(np.dot(u_tweets, u_weights)), axis = 1)
    
    return np.squeeze(att_weights)

def tokens2words(rev_vocabulary, translated_tweet):
    """
    Maps a tokenized tweet back to words and full string using
    a reversed vocabulary returned by create_vocab in module create_vocab_morten
    """
    
    #Remove padding
    translated_tweet = translated_tweet[translated_tweet > 0]
    
    tokens = []

    for idx, value in enumerate(translated_tweet):
        tokens.append(rev_vocabulary[value])
    
    tweet = ""
    
    for entry in tokens:
        tweet = tweet + " {}".format(entry)
    return tokens, tweet
    
    
    
    
    
    
    
    
    
    
    
    
    
    