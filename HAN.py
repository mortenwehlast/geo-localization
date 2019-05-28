#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:52:14 2019

@author: Morten
"""
import numpy as np
import json
import pickle


#Data utilities
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import parallel_backend
from sklearn.externals import joblib #To save CV results
from keras.wrappers.scikit_learn import KerasClassifier

from model_utils import *

#%% MODEL PARAMETERS
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


if __name__ == "__main__":
#%% Load data

    path = ""
    
    data             = np.load(path + 'dataset_full.npz')['tokenized_tweet_data']
    users_and_labels = np.load(path + 'dataset_full.npz')['users_and_labels']
    
    users            = users_and_labels['user_id']
    state_labels     = users_and_labels['state']
    
    print('Data Loaded')
    #%% Preprocess data
    
    #Data stored as tensor (# Users, MAX # tweets, Length of tweets)
    NUM_USERS      = len(data)
    MAX_NUM_TWEETS = len(data[0])
    LEN_TWEET      = len(data[0][0])
    
    
    #Create class weights (balanced model)
    #class_weights = class_weight.compute_class_weight('balanced',
    #                                                 np.unique(state_labels),
    #                                                 state_labels)
        
    
    #Convert to Keras standard
    labels = to_categorical(state_labels)
    
    
    #Set seed
    seed = 1337
    np.random.seed(seed)
    
    #Split in test and train+validation
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1)
    
    
    #Save data
    np.savez_compressed('X_train', X_train = X_train)
    np.savez_compressed('y_train', y_train = y_train)
    np.savez_compressed('X_test', X_test = X_test)
    np.savez_compressed('y_test', y_test = y_test)
    
    print('Data splits saved')
#%%
    RESULT_FOLDER = ""
    SAVE_WORD_MODEL = True
    
    word_model_container = [0]
    
    HAN_model = build_full_model(DENSE_UNITS    = DENSE_UNITS, 
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
                                 save_word_model= SAVE_WORD_MODEL)
    
    checkpoint = ModelCheckpoint(filepath= RESULT_FOLDER + 'weights.{epoch:02d}-{val_acc:.2f}.hdf5',
                                 monitor='val_acc',
                                 verbose=0, 
                                 save_best_only= False,
                                 save_weights_only = True,
                                 mode = 'max',
                                 period = 1)
    
    es = EarlyStopping(monitor='val_acc', 
                       mode='max', 
                       verbose = 1,
                       patience= 3)
    
    print('Model configured. Commencing training.')
    
    history = HAN_model.fit(X_train, y_train,
                              epochs = EPOCHS, 
                              batch_size = BATCH_SIZE,
                              callbacks=[checkpoint, es],
                              validation_split = 0.1,
                              verbose = 1)
    
    print('Model trained. Saving word model and final model.')
    #%% Save model weights and history
    word_model = word_model_container[0]
    word_model.save_weights(RESULT_FOLDER + 'final_word_model_weights.h5')
    history.model.save_weights(RESULT_FOLDER + 'final_model_weights.h5')
    with open(RESULT_FOLDER + 'training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
        
    # Generalization
    
    print('Computing test results.')
    #Predictions
    test_prediction_distributions = history.model.predict(X_test)
    
    #Clean up predictions and true labels
    test_predictions_argmax       = np.argmax(test_prediction_distributions, axis = 1)
    test_true_labels              = np.argmax(y_test, axis = 1)
    
    #Convert to int
    test_predictions_argmax       = [int(item) for item in test_predictions_argmax]
    test_true_labels       = [int(item) for item in test_true_labels]
    
    #Zip predictions and true labels
    pred_true_label_paired_list   = [item for item in zip(test_predictions_argmax, test_true_labels)]
    
    #Generalization to file
    results = {'prediction_distribution': test_prediction_distributions.tolist(), 
               'pred_true_label_paired_list': pred_true_label_paired_list
               }
    
    print('Saving test results.')
    with open(RESULT_FOLDER + 'results.json', 'w') as file:
        json.dump(results, file)
    