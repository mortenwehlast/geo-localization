#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:35:55 2019

@author: Morten
"""

import json
import numpy as np
from global_variables import SPECIAL_TOKENS




word_counts_sorted_path = "word_counts_sorted.npz"

def create_vocab(word_counts_sorted_path = word_counts_sorted_path, max_words = 10000) :
    """ 
    Creates a vocabulary from the most popular words in the user tweets. 
    max_word denotes vocabulary size.
    Returns two vocabularies in order to easily convert tokens back to words:
        words -> index
        index -> word
    """
    sorted_words = np.load(word_counts_sorted_path)['word_counts_sorted']
    vocabulary = {}
    reverse_vocabulary = {}
    
    for idx, token in enumerate(SPECIAL_TOKENS, 1): #Add special tokens in beginning
        vocabulary[token] = idx
        reverse_vocabulary[idx] = token
    
    for idx, entry in enumerate(sorted_words, len(SPECIAL_TOKENS) + 1): #Start after special tokens
        if idx > max_words:
            break
        word, count = entry
        word = word.decode('utf-8') #Python 2 artifact. Convert from bytes to UTF-8
        vocabulary[word] = idx
        reverse_vocabulary[idx] = word
    
    print('Vocabulary Created! # of Words: {}'.format(max_words))
    return vocabulary, reverse_vocabulary
        
        
