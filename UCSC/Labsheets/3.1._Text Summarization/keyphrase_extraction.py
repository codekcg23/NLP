# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: arw
"""

from nltk.corpus import gutenberg
from normalization import normalize_corpus
import nltk
from operator import itemgetter

# Load the Alice corpus from NLTK
alice = gutenberg.sents(fileids='carroll-alice.txt')
alice = [' '.join(ts) for ts in alice]
normalized_alice = filter(None, normalize_corpus(alice, lemmatize=False))
norm_alice = list(normalized_alice)
# Print the first line of the corpus
print(norm_alice[0])

# Create a single long string out of the corpus
def flatten_corpus(corpus):
    return ' '.join([document.strip() 
                     for document in corpus])
                     
# The zip function below 'merges' a set of n lists item by item
# Try to setup 2 lists of the same length and call zip with the two lists as arguments
# Hint: need to cast result to list for viewing contents

# See the effect of the following function by calling compute_ngrams([1,2,3,4], 2) and compute_ngrams([1,2,3,4], 3)
def compute_ngrams(sequence, n):
    return zip(*[sequence[index:] 
                 for index in range(n)])


# We generalize these ideas to get a generic function to get the top n-grams from a corpus
def get_top_ngrams(corpus, ngram_val=1, limit=5):

    corpus = flatten_corpus(corpus)
    tokens = nltk.word_tokenize(corpus)

    # Compute the frequencies of the n-grams using NLTK's FreqDist class
    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(), 
                              key=itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq) 
                     for text, freq in sorted_ngrams]

    return sorted_ngrams   
    
# Now try this function for bigrams
get_top_ngrams(corpus=norm_alice, ngram_val=2,
               limit=10)
# And for bigrams
get_top_ngrams(corpus=norm_alice, ngram_val=3,
               limit=10)

# NLTK has built-in collocation finders can use frequencies of pointwise mutual information (pmi)
# Read and understand the intuitive meaning of PMI from the web
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures

finder = BigramCollocationFinder.from_documents([item.split() 
                                                for item 
                                                in norm_alice])
bigram_measures = BigramAssocMeasures() 

# Using raw frequencies for collocations                                          
finder.nbest(bigram_measures.raw_freq, 10)
# Using mutual information scores for collocations
finder.nbest(bigram_measures.pmi, 10)   


# We can repeat the above for trigrams too
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures

finder = TrigramCollocationFinder.from_documents([item.split() 
                                                for item 
                                                in norm_alice])
trigram_measures = TrigramAssocMeasures()                                                
finder.nbest(trigram_measures.raw_freq, 10)
finder.nbest(trigram_measures.pmi, 10)  





# We now use weighted tag based phrase extraction
# For this, we need to be able to perform some chunking

toy_text = """
Elephants are large mammals of the family Elephantidae 
and the order Proboscidea. Two species are traditionally recognised, 
the African elephant and the Asian elephant. Elephants are scattered 
throughout sub-Saharan Africa, South Asia, and Southeast Asia. Male 
African elephants are the largest extant terrestrial animals. All 
elephants have a long trunk used for many purposes, 
particularly breathing, lifting water and grasping objects. Their 
incisors grow into tusks, which can serve as weapons and as tools 
for moving objects and digging. Elephants' large ear flaps help 
to control their body temperature. Their pillar-like legs can 
carry their great weight. African elephants have larger ears 
and concave backs while Asian elephants have smaller ears 
and convex or level backs.  
"""

from normalization import parse_document
import itertools
from normalization import stopword_list
from gensim import corpora, models

# Extract chunks we are interested in (and omit chinks we are not interested in)
# This process depends on the POS tags and grammar tags used in the corpus we want to use
# Here anything with a chunk tag 'O' is a chink
def get_chunks(sentences, grammar = r'NP: {<DT>? <JJ>* <NN.*>+}'):
    # Build a chunker based on grammar pattern above
    all_chunks = []
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    
    for sentence in sentences:
        # POS tag the sentences
        tagged_sents = nltk.pos_tag_sents(
                            [nltk.word_tokenize(sentence)])
        # Extract the chunks
        chunks = [chunker.parse(tagged_sent) 
                  for tagged_sent in tagged_sents]
        # Get word, pos tag, chunk tag triples
        wtc_sents = [nltk.chunk.tree2conlltags(chunk)
                     for chunk in chunks]    
         
        flattened_chunks = list(
                            itertools.chain.from_iterable(
                                wtc_sent for wtc_sent in wtc_sents)
                           )
        # Get only valid chunks based on tags
        valid_chunks_tagged = [(status, [wtc for wtc in chunk]) 
                        for status, chunk 
                        in itertools.groupby(flattened_chunks, 
                                             # get only if chunk != 'O'
                                             lambda wdposchnk: wdposchnk[2] != 'O')]
        # Append words in each chunk to make phrases
        valid_chunks = [' '.join(word.lower() 
                                for word, tag, chunk 
                                in wtc_group 
                                    if word.lower() 
                                        not in stopword_list) 
                                    for status, wtc_group 
                                    in valid_chunks_tagged
                                        if status]
        # Append all valid chunked phrases                                    
        all_chunks.append(valid_chunks)
    
    return all_chunks
    
sentences = parse_document(toy_text)          
valid_chunks = get_chunks(sentences)
# Print all valid chunks
print(valid_chunks)


# Build a chunk extractor based on TF-IDF weights instead of frequencies
def get_tfidf_weighted_keyphrases(sentences, 
                                  grammar=r'NP: {<DT>? <JJ>* <NN.*>+}',
                                  top_n=10):
    # Get valid chunks as before
    valid_chunks = get_chunks(sentences, grammar=grammar)
    # This time build a tf-idf based model                                 
    dictionary = corpora.Dictionary(valid_chunks)
    corpus = [dictionary.doc2bow(chunk) for chunk in valid_chunks]
    
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # Get phrases and their tf-idf weights
    weighted_phrases = {dictionary.get(id): round(value,3) 
                        for doc in corpus_tfidf 
                        for id, value in doc}
                            
    weighted_phrases = sorted(weighted_phrases.items(), 
                              key=itemgetter(1), reverse=True)
    # Return the top n weighted phrases
    return weighted_phrases[:top_n]

# Get top 10 tf-idf weighted keyphrases for toy_text
get_tfidf_weighted_keyphrases(sentences, top_n=10)

# Try with other corpora such as the Alice corpus from NLTK's Guttenburg collection
get_tfidf_weighted_keyphrases(alice, top_n=10)
    



