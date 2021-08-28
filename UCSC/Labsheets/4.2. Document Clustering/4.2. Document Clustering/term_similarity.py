# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

import numpy as np
from scipy.stats import itemfreq

# Simplest representation of tokens is as a sequence of characters
def vectorize_terms(terms):
    terms = [term.lower() for term in terms]
    terms = [np.array(list(term)) for term in terms]
    terms = [np.array([ord(char) for char in term]) 
                for term in terms]
    return terms
    
# Another possibility is a bag of characters representation (without order constraint)
# We just use the frequency of each character as our word/token representation
def boc_term_vectors(word_list):
    word_list = [word.lower() for word in word_list]
    unique_chars = np.unique(
                        np.hstack([list(word) 
                        for word in word_list]))
    word_list_term_counts = [{char: count for char, count in itemfreq(list(word))}
                             for word in word_list]
    
    boc_vectors = [np.array([int(word_term_counts.get(char, 0)) 
                            for char in unique_chars])
                   for word_term_counts in word_list_term_counts]
    return list(unique_chars), boc_vectors
                             
                             
root = 'Believe'
term1 = 'beleive'
term2 = 'bargain'
term3 = 'Elephant'    

terms = [root, term1, term2, term3]

# Vectorize the root and the terms
vec_root, vec_term1, vec_term2, vec_term3 = vectorize_terms(terms)

# Show vector representations
print('''
root: {}
term1: {}
term2: {}
term3: {}
'''.format(vec_root, vec_term1, vec_term2, vec_term3))

# Vectorize root and terms in bag-of-characters representations
features, (boc_root, boc_term1, boc_term2, boc_term3) = boc_term_vectors(terms)
# Show the full list of features (all characters of all terms)
print('Features:', features)
# Show vector representation
print('''
root: {}
term1: {}
term2: {}
term3: {}
'''.format(boc_root, boc_term1, boc_term2, boc_term3))


# Catch the 'unequal length terms' exception for the metrics that rely on equal length terms
def hamming_distance(u, v, norm=False):
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    return (u != v).sum() if not norm else (u != v).mean()
    
def manhattan_distance(u, v, norm=False):
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    return abs(u - v).sum() if not norm else abs(u - v).mean()

def euclidean_distance(u,v):
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    distance = np.sqrt(np.sum(np.square(u - v)))
    return distance

import copy
import pandas as pd

def levenshtein_edit_distance(u, v):
    # Convert to lower case
    u = u.lower()
    v = v.lower()
    # Base cases
    if u == v: return 0
    elif len(u) == 0: return len(v)
    elif len(v) == 0: return len(u)
    # Initialize edit distance matrix
    edit_matrix = []
    # Initialize two distance matrices 
    du = [0] * (len(v) + 1)
    dv = [0] * (len(v) + 1)
    # du: the previous row of edit distances
    for i in range(len(du)):
        du[i] = i
    # dv : the current row of edit distances    
    for i in range(len(u)):
        dv[0] = i + 1
        # Compute cost as per algorithm
        for j in range(len(v)):
            cost = 0 if u[i] == v[j] else 1
            dv[j + 1] = min(dv[j] + 1, du[j + 1] + 1, du[j] + cost)
        # Assign dv to du for next iteration
        for j in range(len(du)):
            du[j] = dv[j]
        # Copy dv to the edit matrix
        edit_matrix.append(copy.copy(dv))
    # Compute the final edit distance and edit matrix    
    distance = dv[len(v)]
    edit_matrix = np.array(edit_matrix)
    edit_matrix = edit_matrix.T
    edit_matrix = edit_matrix[1:,]
    edit_matrix = pd.DataFrame(data=edit_matrix,
                               index=list(v),
                               columns=list(u))
    return distance, edit_matrix
    
def cosine_distance(u, v):
    distance = 1.0 - (np.dot(u, v) / 
                        (np.sqrt(sum(np.square(u))) * np.sqrt(sum(np.square(v))))
                     )
    return distance



# Set up the term vectors     
root_term = root
root_vector = vec_root
root_boc_vector = boc_root

terms = [term1, term2, term3]
vector_terms = [vec_term1, vec_term2, vec_term3]
boc_vector_terms = [boc_term1, boc_term2, boc_term3]


# HAMMING DISTANCE - will give error for unequal length terms
# for term, vector_term in zip(terms, vector_terms):
#     print('Hamming distance between root: {} and term: {} is {}'.format(root_term,
#                                                                 term,
#                                                                 hamming_distance(root_vector, vector_term, norm=False)))


# for term, vector_term in zip(terms, vector_terms):
#     print('Normalized Hamming distance between root: {} and term: {} is {}'.format(root_term,
#                                                                 term,
#                                                                 round(hamming_distance(root_vector, vector_term, norm=True), 2)))


# # MANHATTAN DISTANCE - will give error for unequal length terms
# for term, vector_term in zip(terms, vector_terms):
#     print('Manhattan distance between root: {} and term: {} is {}'.format(root_term,
#                                                                 term,
#                                                                 manhattan_distance(root_vector, vector_term, norm=False)))

# for term, vector_term in zip(terms, vector_terms):
#     print('Normalized Manhattan distance between root: {} and term: {} is {}'.format(root_term,
#                                                                 term,
#                                                                 round(manhattan_distance(root_vector, vector_term, norm=True),2)))


# # EUCLIDEAN DISTANCE - will give error for unequal length terms
# for term, vector_term in zip(terms, vector_terms):
#     print('Euclidean distance between root: {} and term: {} is {}'.format(root_term,
#                                                                 term,
#                                                                 round(euclidean_distance(root_vector, vector_term),2)))


# LEVENSHTEIN EDIT DISTANCE - doesn't depend on lengths of terms being equal (why?)
for term in terms:
    edit_d, edit_m = levenshtein_edit_distance(root_term, term)
    print('Computing distance between root: {} and term: {}'.format(root_term,
                                                                    term))
    print('Levenshtein edit distance is {}'.format(edit_d))
    print('The complete edit distance matrix is depicted below')
    print(edit_m)
    print('-'*30)                                                                           


# COSINE DISTANCE\SIMILARITY - doesn't depend on lengths of terms being equal
for term, boc_term in zip(terms, boc_vector_terms):
    print('Analyzing similarity between root: {} and term: {}'.format(root_term,
                                                                      term))
    distance = round(cosine_distance(root_boc_vector, boc_term),2)
    similarity = 1 - distance                                                           
    print('Cosine distance  is {}'.format(distance))
    print('Cosine similarity  is {}'.format(similarity))
    print('-'*40)
                                                                

