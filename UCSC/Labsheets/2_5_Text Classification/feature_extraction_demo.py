# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

# Let's start with a 'toy' corpus
CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]

# We use new_doc as our test dataset
new_doc = ['loving this blue sky today']

import pandas as pd

def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
                      columns=feature_names)
    print(df)

# We pass our CORPUS to the simplest bow extractor we created
from feature_extractors import bow_extractor    
bow_vectorizer, bow_features = bow_extractor(CORPUS)
features = bow_features.todense() # Since we can't view the default 'sparse matrix'
print(features)

# Remember, we always need to extract the same features from our test data too!
new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features = new_doc_features.todense()
print(new_doc_features)

# Let's see which words/tokens these counts are for...
feature_names = bow_vectorizer.get_feature_names()
print(feature_names)

# Let's print both the feature names and counts together
# - first for the training data and then for the test data
display_features(features, feature_names)
display_features(new_doc_features, feature_names)


# Now let's try the same with tf-idf instead of frequency counts
# We use the tfidf_transformer function we defined
import numpy as np
from feature_extractors import tfidf_transformer
feature_names = bow_vectorizer.get_feature_names()

# We again convert to the dense form to print the values out    
tfidf_trans, tdidf_features = tfidf_transformer(bow_features)
features = np.round(tdidf_features.todense(), 2)
display_features(features, feature_names)
# We do the same for the test document
nd_tfidf = tfidf_trans.transform(new_doc_features)
nd_features = np.round(nd_tfidf.todense(), 2)
display_features(nd_features, feature_names)



# We can also compute tf-idf scores/vectors ourselves from scratch
# - without using sklearn's TfidfTransformer class
import scipy.sparse as sp
from numpy.linalg import norm
feature_names = bow_vectorizer.get_feature_names()

# We compute term frequencies by simply using our bow model
tf = bow_features.todense()
tf = np.array(tf, dtype='float64')

# Check if our term frequencies are as expected
display_features(tf, feature_names)

# We next build the document frequency matrix
df = np.diff(sp.csc_matrix(bow_features, copy=True).indptr)
df = 1 + df # to smoothen idf later

# Check if our document frequencies are as expected
display_features([df], feature_names)

# Now compute the inverse document frequencies
total_docs = 1 + len(CORPUS)
idf = 1.0 + np.log(float(total_docs) / df)

# Are our inverse document frequencies what we expected?
display_features([np.round(idf, 2)], feature_names)

# Now compute the idf diagonal matrix  
total_features = bow_features.shape[1]
idf_diag = sp.spdiags(idf, diags=0, m=total_features, n=total_features)
idf = idf_diag.todense()

# Is the idf diagonal matrix as expected?
print(np.round(idf, 2))

# Now compute the full tfidf feature matrix
tfidf = tf * idf

# Is the tfidf feature matrix what we expected?
display_features(np.round(tfidf, 2), feature_names)

# Now compute the L2 norms 
norms = norm(tfidf, axis=1)

# Display the L2 norms for each document
print(np.round(norms, 2))

# Now compute the 'normalized' tfidf
norm_tfidf = tfidf / norms[:, None]

# Check if the final tfidf feature matrix is as expected
# Is it the same as what we got using the TfidfTransformer class of sklearn?
display_features(np.round(norm_tfidf, 2), feature_names)
 

# Now do the same for the test data 
# First, compute the term freqs from bow freqs for the test data - new_doc
nd_tf = new_doc_features
nd_tf = np.array(nd_tf, dtype='float64')

# Next compute tfidf using idf matrix from the train corpus
nd_tfidf = nd_tf*idf
nd_norms = norm(nd_tfidf, axis=1)
norm_nd_tfidf = nd_tfidf / nd_norms[:, None]

# Check the new_doc tfidf feature vector
display_features(np.round(norm_nd_tfidf, 2), feature_names)



# sklearn's TfidfVectorizer provides a transformer to extract tfidf scores directly
# from raw data - avoiding the need for CountVectorizer based bow scores
from feature_extractors import tfidf_extractor
    
tfidf_vectorizer, tdidf_features = tfidf_extractor(CORPUS)
display_features(np.round(tdidf_features.todense(), 2), feature_names)

nd_tfidf = tfidf_vectorizer.transform(new_doc)
display_features(np.round(nd_tfidf.todense(), 2), feature_names)    



# We can also do more sophisticated word-vector models using Google's word2vec algorithm
# using the gensim python package
import gensim
import nltk

TOKENIZED_CORPUS = [nltk.word_tokenize(sentence) 
                    for sentence in CORPUS]
tokenized_new_doc = [nltk.word_tokenize(sentence) 
                    for sentence in new_doc]                        

# Model parameters for the NN-based word2vec 'word embeddings':
# size - dimension of the word vectors (tens to thousands)
# window - window size to conside the context of a word
# min_count - the minimum frequency of a word in the whole corpus to be included in vocabulary
# sample - used to downsample the effects of the occurence of frequent words
model = gensim.models.Word2Vec(TOKENIZED_CORPUS, 
                               vector_size=10,
                               window=10,
                               min_count=2,
                               sample=1e-3)

# Averaging word vectors of a document
from feature_extractors import averaged_word_vectorizer

avg_word_vec_features = averaged_word_vectorizer(corpus=TOKENIZED_CORPUS,
                                                 model=model.wv,
                                                 num_features=10)
print(np.round(avg_word_vec_features, 3))

nd_avg_word_vec_features = averaged_word_vectorizer(corpus=tokenized_new_doc,
                                                    model=model.wv,
                                                    num_features=10)
print(np.round(nd_avg_word_vec_features, 3))


# Using tfidf weighted average of word vectors in a document              
from feature_extractors import tfidf_weighted_averaged_word_vectorizer

corpus_tfidf = tdidf_features
vocab = tfidf_vectorizer.vocabulary_
wt_tfidf_word_vec_features = tfidf_weighted_averaged_word_vectorizer(corpus=TOKENIZED_CORPUS,
                                                                     tfidf_vectors=corpus_tfidf,
                                                                     tfidf_vocabulary=vocab,
                                                                     model=model.wv, 
                                                                     num_features=10)
print(np.round(wt_tfidf_word_vec_features, 3))

nd_wt_tfidf_word_vec_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_new_doc,
                                                                     tfidf_vectors=nd_tfidf,
                                                                     tfidf_vocabulary=vocab,
                                                                     model=model.wv, 
                                                                     num_features=10)
print(np.round(nd_wt_tfidf_word_vec_features, 3))


                                                                 