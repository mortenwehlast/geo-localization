#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:29:26 2019

@author: Morten

Extracts the most and the least confident prediction for the balanced and unbalanced models 
and visualizes the distributions of attention weights on word and tweet level.

"u" denotes unbalanced
"b" denotes balanced

"""

#%%
import numpy as np
import json
import pickle

from tokenizer import tokenize
from sentence_tokenizer import SentenceTokenizer
from create_vocab_morten import create_vocab
import matplotlib.pyplot as plt

from model_utils import *

u_results_path = "ub_results.json"
b_results_path = "results.json"

#Get results
with open(u_results_path, 'rb') as file:
    u_results = json.load(file)
    
with open(b_results_path, 'rb') as file:
    b_results = json.load(file)


u_dists = u_results['prediction_distribution']
u_preds = u_results['pred_true_label_paired_list']
b_dists = b_results['prediction_distribution']
b_preds = b_results['pred_true_label_paired_list']



#%%

u_most_conf_idx = 0
b_most_conf_idx = 0
u_least_conf_idx = 0
b_least_conf_idx = 0


#Find most confident predicitions to visualize
u_most_conf_pred = 0
for idx, dist in enumerate(u_dists):
    dist_max = max(dist)
    if dist_max > u_most_conf_pred:
        u_most_conf_pred  = dist_max
        u_most_conf_idx = idx
        
b_most_conf_pred = 0      
for idx, dist in enumerate(b_dists):
    dist_max = max(dist)
    if dist_max > b_most_conf_pred:
        b_most_conf_pred  = dist_max
        b_most_conf_idx = idx

#Find least confident predictions
u_least_conf_pred = 99999
for idx, dist in enumerate(u_dists):
    dist_max = max(dist)
    if dist_max < u_least_conf_pred:
        u_least_conf_pred  = dist_max
        u_least_conf_idx = idx
        
b_least_conf_pred = 99999     
for idx, dist in enumerate(b_dists):
    dist_max = max(dist)
    if dist_max < b_least_conf_pred:
        b_least_conf_pred  = dist_max
        b_least_conf_idx = idx



#Extract the most confident predictions
u_max_pred = u_preds[u_most_conf_idx]
b_max_pred = b_preds[b_most_conf_idx]

#Extract the least confident predictions
u_min_pred = u_preds[u_least_conf_idx]
b_min_pred = b_preds[b_least_conf_idx]


#%%
#Load test data to visualize attention
test_data_path = "X_test.npz"
test_data = np.load(test_data_path)['X_test']

#Extract observations
u_max_conf_obs = test_data[u_most_conf_idx]
b_max_conf_obs = test_data[b_most_conf_idx]
u_min_conf_obs = test_data[u_least_conf_idx]
b_min_conf_obs = test_data[b_least_conf_idx]

#Get tweet attention maps
w_u_model_path = "ub_final_model_weights.h5"
u_model = build_model()
u_model.load_weights(w_u_model_path)

w_b_model_path = "final_model_weights.h5" 
b_model = build_model()
b_model.load_weights(w_b_model_path)

#Get tweet level attention
u_max_tweet_att = attention_map_tweet(u_max_conf_obs, u_model)
b_max_tweet_att = attention_map_tweet(b_max_conf_obs, b_model)
u_min_tweet_att = attention_map_tweet(u_min_conf_obs, u_model)
b_min_tweet_att = attention_map_tweet(b_min_conf_obs, b_model)


#Get tweet which is paid most attention to
u_max_tweet_most_att_idx = u_max_tweet_att.argsort()[-1:][0]
b_max_tweet_most_att_idx = b_max_tweet_att.argsort()[-1:][0]
u_min_tweet_most_att_idx = u_min_tweet_att.argsort()[-1:][0]
b_min_tweet_most_att_idx = b_min_tweet_att.argsort()[-1:][0]

u_max_tweet_most_att = u_max_conf_obs[u_max_tweet_most_att_idx]
b_max_tweet_most_att = b_max_conf_obs[b_max_tweet_most_att_idx]
u_min_tweet_most_att = u_min_conf_obs[u_min_tweet_most_att_idx]
b_min_tweet_most_att = b_min_conf_obs[b_min_tweet_most_att_idx]

#Get word attention maps
w_u_word_model_path = "ub_final_word_model_weights.h5"
u_word_model = build_word_model()
u_word_model.load_weights(w_u_word_model_path)

w_b_word_model_path = "final_word_model_weights.h5"
b_word_model = build_word_model()
b_word_model.load_weights(w_b_word_model_path)

#Get word level attention for most important tweet
u_max_word_att = attention_map_word(u_max_tweet_most_att, u_word_model)
b_max_word_att = attention_map_word(b_max_tweet_most_att, b_word_model)
u_min_word_att = attention_map_word(u_min_tweet_most_att, u_word_model)
b_min_word_att = attention_map_word(b_min_tweet_most_att, b_word_model)

#%%

#Get reverse vocab
vocab, rev_vocab = create_vocab(max_words = 50000)

#Reverse tweets
u_max_tokens, u_max_tweet = tokens2words(rev_vocab, u_max_tweet_most_att)
b_max_tokens, b_max_tweet = tokens2words(rev_vocab, b_max_tweet_most_att)
u_min_tokens, u_min_tweet = tokens2words(rev_vocab, u_min_tweet_most_att)
b_min_tokens, b_min_tweet = tokens2words(rev_vocab, b_min_tweet_most_att)


#%% Create plots

#Prediction distributions most and least confident
plt.bar(np.arange(0, 49), u_dists[u_most_conf_idx])
plt.xlabel('state')
plt.ylabel('probability')
plt.title('Probability distribution for most confident prediction (unbalanced)')
plt.savefig('u_dist_most_conf')
plt.show()

plt.bar(np.arange(0, 49), u_dists[u_least_conf_idx])
plt.xlabel('state')
plt.ylabel('probability')
plt.title('Probability distribution for least confident prediction (unbalanced)')
plt.savefig('u_dist_least_conf')
plt.show()




#Prediction distributions most and least confident
plt.bar(np.arange(0, 49), b_dists[b_most_conf_idx])
plt.xlabel('state')
plt.ylabel('probability')
plt.title('Probability distribution for most confident prediction (balanced)')
plt.savefig('b_dist_most_conf')
plt.show()


plt.bar(np.arange(0, 49), b_dists[b_least_conf_idx])
plt.xlabel('state')
plt.ylabel('probability')
plt.title('Probability distribution for least confident prediction (balanced)')
plt.savefig('b_dist_least_conf')
plt.show()




#Word level attention plots

#Unbalaned most confident
plt.bar(np.arange(1,31), u_max_word_att)
plt.xlabel('word position')
plt.ylabel('attention')
plt.title('Word attention for most confident prediction (unbalanced)')
plt.savefig('u_att_most_conf')
plt.show()

#Unbalanced least confident
plt.bar(np.arange(1,31), u_min_word_att)
plt.xlabel('word position')
plt.ylabel('attention')
plt.title('Word attention for least confident prediction (unbalanced)')
plt.savefig('u_att_least_conf')
plt.show()


#Balanced most confident
plt.bar(np.arange(1,31), b_max_word_att)
plt.xlabel('word position')
plt.ylabel('attention')
plt.title('Word attention for most confident prediction (balanced)')
plt.savefig('b_att_most_conf')
plt.show()

#Balanced least confident
plt.bar(np.arange(1,31), b_min_word_att)
plt.xlabel('word position')
plt.ylabel('attention')
plt.title('Word attention for least confident prediction (balanced)')
plt.savefig('b_att_least_conf')
plt.show()