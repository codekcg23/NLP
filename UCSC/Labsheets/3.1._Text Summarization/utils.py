# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

# We define a generic feature extraction function for our work from what we did before
# NB: It returns the particular 'vectorizer' used as well as the extracted feature matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_feature_matrix(documents, feature_type='frequency'):

    feature_type = feature_type.lower().strip()  
    
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=1, 
                                     ngram_range=(1, 1))
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=1, 
                                     ngram_range=(1, 1))
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=1, 
                                     ngram_range=(1, 1))
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    
    return vectorizer, feature_matrix


# For our work we use the SVD implementation in scipy
from scipy.sparse.linalg import svds
    
def low_rank_svd(matrix, singular_count=2):
    
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt
