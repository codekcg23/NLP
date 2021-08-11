# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

# Understanding Machine Learning Evaluation metrics using a 'confusion matrix'
# We use sklearn's metrics method and compare results with what we expect by calculating
# the metrics manually

from sklearn import metrics
import numpy as np
import pandas as pd
from collections import Counter

# First some dummy data for the metric methods to compute the metrics
actual_labels = ['spam', 'ham', 'spam', 'spam', 'spam',
               'ham', 'ham', 'spam', 'ham', 'spam',
               'spam', 'ham', 'ham', 'ham', 'spam',
               'ham', 'ham', 'spam', 'spam', 'ham']
              
predicted_labels = ['spam', 'spam', 'spam', 'ham', 'spam',
                    'spam', 'ham', 'ham', 'spam', 'spam',
                    'ham', 'ham', 'spam', 'ham', 'ham',
                    'ham', 'spam', 'ham', 'spam', 'spam']

# Computer the cells of the confusion matrix                    
ac = Counter(actual_labels)                     
pc = Counter(predicted_labels)  

print('Actual counts:', ac.most_common())
print('Predicted counts:', pc.most_common())         
        
# Define confusion matrix using sklearn function
cm = metrics.confusion_matrix(y_true=actual_labels,
                         y_pred=predicted_labels,
                         labels=['spam','ham'])
print(pd.DataFrame(data=cm, 
                   columns=pd.MultiIndex(levels=[['Predicted:'],
                                                 ['spam','ham']], 
                                         codes=[[0,0],[0,1]]), 
                   index=pd.MultiIndex(levels=[['Actual:'],
                                               ['spam','ham']], 
                                       codes=[[0,0],[0,1]])))

# We need to declare what is positive - other will be taken as negative                                      
positive_class = 'spam'

# We first calculate the metric 'accuracy' using metric function
accuracy = np.round(
                metrics.accuracy_score(y_true=actual_labels,
                                       y_pred=predicted_labels),2)

# Now we calculate 'accuracy' metric manually
# For this, we need to declare the counts in each cell... manually!
true_positive = 5.
false_positive = 6.
false_negative = 5.
true_negative = 4.

accuracy_manual = np.round(
                    (true_positive + true_negative) /
                      (true_positive + true_negative +
                       false_negative + false_positive),2)
print('Accuracy:', accuracy)
print('Manually computed accuracy:', accuracy_manual)


# We calculate 'precision' using the function and compare with our manual calculation
precision = np.round(
                metrics.precision_score(y_true=actual_labels,
                                        y_pred=predicted_labels,
                                        pos_label=positive_class),2)
precision_manual = np.round(
                        (true_positive) /
                        (true_positive + false_positive),2)
print('Precision:', precision)
print('Manually computed precision:', precision_manual)


# We calculate 'recall' using the function and compare with our manual calculation
recall = np.round(
            metrics.recall_score(y_true=actual_labels,
                                 y_pred=predicted_labels,
                                 pos_label=positive_class),2)
recall_manual = np.round(
                    (true_positive) /
                    (true_positive + false_negative),2)
print('Recall:', recall)
print('Manually computed recall:', recall_manual)


# Finally we calculate 'F1-score' using the function and compare with our manual calculation
f1_score = np.round(
                metrics.f1_score(y_true=actual_labels,
                                 y_pred=predicted_labels,
                                 pos_label=positive_class),2) 
f1_score_manual = np.round(
                    (2 * precision * recall) /
                    (precision + recall),2)
print('F1 score:', f1_score)
print('Manually computed F1 score:', f1_score_manual)

