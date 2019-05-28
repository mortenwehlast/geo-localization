#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:40:56 2019

@author: Morten

Calculate performance scores for both the unbalanced and the balanced model.

"""

import json
import numpy as np

unbalanced_path = "/Users/Morten/Library/Mobile Documents/com~apple~CloudDocs/Studie/6. semester/Bachelor/results/unbalanced/ub_results.json"
balanced_path   = "/Users/Morten/Library/Mobile Documents/com~apple~CloudDocs/Studie/6. semester/Bachelor/results/balanced_with_class_weights/results.json"

with open(unbalanced_path, 'rb') as file:
    unbalanced_dict = json.load(file)
    
with open(balanced_path, 'rb') as file:
    balanced_dict = json.load(file)

#Unbalanced data
unbalanced_distributions = unbalanced_dict['prediction_distribution']
unbalanced_predictions   = unbalanced_dict['pred_true_label_paired_list']

#Balanced data
balanced_distributions   = balanced_dict['prediction_distribution']
balanced_predictions     = balanced_dict['pred_true_label_paired_list']

#N observations
n_obs = len(unbalanced_predictions)

#Exact match
unbalanced_exact = sum([entry[0] == entry[1] for entry in unbalanced_predictions]) / n_obs
balanced_exact   = sum([entry[0] == entry[1] for entry in balanced_predictions]) / n_obs

unbalanced_top2_sum = 0
unbalanced_top3_sum = 0
unbalanced_top5_sum = 0

balanced_top2_sum   = 0
balanced_top3_sum   = 0
balanced_top5_sum   = 0


for i in range(len(unbalanced_predictions)):
    true_label = unbalanced_predictions[i][1]
    
    unbalanced_top2_sum += 1 if true_label in np.asarray(unbalanced_distributions[i]).argsort()[-2:] else 0
    unbalanced_top3_sum += 1 if true_label in np.asarray(unbalanced_distributions[i]).argsort()[-3:] else 0
    unbalanced_top5_sum += 1 if true_label in np.asarray(unbalanced_distributions[i]).argsort()[-5:] else 0
    balanced_top2_sum   += 1 if true_label in np.asarray(balanced_distributions[i]).argsort()[-2:] else 0
    balanced_top3_sum   += 1 if true_label in np.asarray(balanced_distributions[i]).argsort()[-3:] else 0
    balanced_top5_sum   += 1 if true_label in np.asarray(balanced_distributions[i]).argsort()[-5:] else 0


    
#Calc accuracies
unbalanced_top2_acc = unbalanced_top2_sum / n_obs
unbalanced_top3_acc = unbalanced_top3_sum / n_obs
unbalanced_top5_acc = unbalanced_top5_sum / n_obs
balanced_top2_acc   = balanced_top2_sum / n_obs
balanced_top3_acc   = balanced_top3_sum / n_obs
balanced_top5_acc   = balanced_top5_sum / n_obs

#%%

print("Accuracies on all four metrics\n \
      (unbalanced, balanced)\n \
      Exact: ({0} , {1})\n \
      Top2 : ({2} , {3})\n \
      Top3 : ({4} , {5})\n \
      Top5 : ({6} , {7})".format(unbalanced_exact, balanced_exact, 
                                 unbalanced_top2_acc, balanced_top2_acc, 
                                 unbalanced_top3_acc, balanced_top3_acc, 
                                 unbalanced_top5_acc, balanced_top5_acc))


#%% Plots


un_hist_path = "/Users/Morten/Library/Mobile Documents/com~apple~CloudDocs/Studie/6. semester/Bachelor/results/unbalanced/ub_training_history.pkl"
ba_hist_path = "/Users/Morten/Library/Mobile Documents/com~apple~CloudDocs/Studie/6. semester/Bachelor/results/balanced_with_class_weights/training_history.pkl"

with open(un_hist_path, 'rb') as pickle_file:
    unbalanced_history = pickle.load(pickle_file)
    
with open(ba_hist_path, 'rb') as pickle_file:
    balanced_history   = pickle.load(pickle_file)


plt.plot(np.arange(1,6),unbalanced_history['acc'])
plt.plot(np.arange(1,6),unbalanced_history['val_acc'])
plt.title('Model Accuracy (Unbalanced)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('unbalanced_accuracy')
plt.show()


plt.plot(np.arange(1,6), balanced_history['acc'])
plt.plot(np.arange(1,6), balanced_history['val_acc'])
plt.title('Model Accuracy (Balanced)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('balanced_accuracy')
plt.show()

plt.plot(np.arange(1,6), unbalanced_history['loss'])
plt.plot(np.arange(1,6), unbalanced_history['val_loss'])
plt.title('Model Loss (Unbalanced)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('unbalanced_loss')
plt.show()

plt.plot(np.arange(1,6), balanced_history['loss'])
plt.plot(np.arange(1,6), balanced_history['val_loss'])
plt.title('Model Loss (Balanced)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('balanced_loss')
plt.show()
