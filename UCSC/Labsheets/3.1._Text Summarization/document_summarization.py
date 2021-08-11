# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

from normalization import normalize_corpus, parse_document
from utils import build_feature_matrix, low_rank_svd
import numpy as np

# We simulate a toy 'document collection' as follows
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

# We define a function to calculate the summarization ratio - with default 0.5
from gensim.summarization import summarize

# gensim uses the popular TextRank algorithm to produce its summaries
def text_summarization_gensim(text, summary_ratio=0.5):
    
    summary = summarize(text, split=True, ratio=summary_ratio)
    for sentence in summary:
        print(sentence)

# Parse the document using code in our normalization.py module
docs = parse_document(toy_text)
text = ' '.join(docs) # 'flatten' the 'corpus' into a singe long string
text_summarization_gensim(text, summary_ratio=0.4) # get a summary which is a little less than half the original



# We will now try to build our own summarization algorithm
# First we check the data we have   
sentences = parse_document(toy_text)
norm_sentences = normalize_corpus(sentences,lemmatize=False) 
total_sentences = len(norm_sentences)
print('Total Sentences in Document:', total_sentences)  

# For LSA, we need to first define the # of sentences we want (n) and the # of topics (k)
num_sentences = 3
num_topics = 2

# We call our utility function to build a feature matrix using CountVectorizer and TfidfVectorizer
# Here we use the bag of words features by passing the parameter 'frequency' - what are the other options?
vec, dt_matrix = build_feature_matrix(sentences, 
                                      feature_type='frequency')
# We need to fist transpose our document-term matrix to a term-document matrix
td_matrix = dt_matrix.transpose()
td_matrix = td_matrix.multiply(td_matrix > 0)

# We need to get low rank SVD components from our utils module
u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  

# We remove singular values that are lower than a heuristic threshold - here 0.5                                       
sv_threshold = 0.5
min_sigma_value = max(s) * sv_threshold
s[s < min_sigma_value] = 0

# Compute salience scores for all sentences in document
salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
print(np.round(salience_scores, 2))

# Rank sentences based on their salience scores
top_sentence_indices = salience_scores.argsort()[-num_sentences:][::-1]
top_sentence_indices.sort()

# View the highest scoring sentence index positions
print(top_sentence_indices)

# Get the document summary by combining the above sentences
# Compare the output with what gensim produces with its TextRank implementation
for index in top_sentence_indices:
    print(sentences[index])
    




# Putting all the above together, we can define a generic LSA based text summarizer as follows
def lsa_text_summarizer(documents, num_sentences=2,
                        num_topics=2, feature_type='frequency',
                        sv_threshold=0.5):
                            
    vec, dt_matrix = build_feature_matrix(documents, 
                                          feature_type=feature_type)

    td_matrix = dt_matrix.transpose()
    td_matrix = td_matrix.multiply(td_matrix > 0)

    u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  
    min_sigma_value = max(s) * sv_threshold
    s[s < min_sigma_value] = 0
    
    salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
    top_sentence_indices = salience_scores.argsort()[-num_sentences:][::-1]
    top_sentence_indices.sort()
    
    for index in top_sentence_indices:
        print(sentences[index])
    
    
    
# Sometimes, we want to be able to visualize our sentence 'graph'
# For this, we can use the popular general purpose networkx package
import networkx

# Define number of sentences in final summary
num_sentences = 3
# Construct weighted document term matrix (using tfidf instead of frequencies as before)
vec, dt_matrix = build_feature_matrix(norm_sentences, 
                                      feature_type='tfidf')

# Construct the document similarity matrix
similarity_matrix = (dt_matrix * dt_matrix.T)
# View the document similarity matrix by making the sparse matric dense!
print(np.round(similarity_matrix.todense(), 2))

# We can now build a similarity graph using the networkx package
similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
# View the network for this simple 9 sentence similarity graph
networkx.draw_networkx(similarity_graph)


# Now compute the pagerank scores for all the sentences
scores = networkx.pagerank(similarity_graph)
# Rank the sentences based on their scores
ranked_sentences = sorted(((score, index) 
                            for index, score 
                            in scores.items()), 
                          reverse=True)
# This is a list, so to view the structure we simply type it in the console
ranked_sentences

# Once again, we get the top sentence indices for our summary
top_sentence_indices = [ranked_sentences[index][1] 
                        for index in range(num_sentences)]
top_sentence_indices.sort()
# We can view the top sentence indices
print(top_sentence_indices)

# And finally construct the document summary to output
for index in top_sentence_indices:
    print(sentences[index])
    

# Putting all the above together, we can define a generic TextRank based text summarizer as follows
def textrank_text_summarizer(documents, num_sentences=2,
                             feature_type='frequency'):
    
    vec, dt_matrix = build_feature_matrix(norm_sentences, 
                                      feature_type='tfidf')
    similarity_matrix = (dt_matrix * dt_matrix.T)
        
    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)   
    
    ranked_sentences = sorted(((score, index) 
                                for index, score 
                                in scores.items()), 
                              reverse=True)

    top_sentence_indices = [ranked_sentences[index][1] 
                            for index in range(num_sentences)]
    top_sentence_indices.sort()
    
    for index in top_sentence_indices:
        print(sentences[index])                            





# Now for a slightly more realistic summarization task
# We use both the above summarizers by calling their reusable functions

DOCUMENT = """
The Elder Scrolls V: Skyrim is an open world action role-playing video game 
developed by Bethesda Game Studios and published by Bethesda Softworks. 
It is the fifth installment in The Elder Scrolls series, following 
The Elder Scrolls IV: Oblivion. Skyrim's main story revolves around 
the player character and their effort to defeat Alduin the World-Eater, 
a dragon who is prophesied to destroy the world. 
The game is set two hundred years after the events of Oblivion 
and takes place in the fictional province of Skyrim. The player completes quests 
and develops the character by improving skills. 
Skyrim continues the open world tradition of its predecessors by allowing the 
player to travel anywhere in the game world at any time, and to 
ignore or postpone the main storyline indefinitely. The player may freely roam 
over the land of Skyrim, which is an open world environment consisting 
of wilderness expanses, dungeons, cities, towns, fortresses and villages. 
Players may navigate the game world more quickly by riding horses, 
or by utilizing a fast-travel system which allows them to warp to previously 
Players have the option to develop their character. At the beginning of the game, 
players create their character by selecting one of several races, 
including humans, orcs, elves and anthropomorphic cat or lizard-like creatures, 
and then customizing their character's appearance.discovered locations. Over the 
course of the game, players improve their character's skills, which are numerical 
representations of their ability in certain areas. There are eighteen skills 
divided evenly among the three schools of combat, magic, and stealth. 
Skyrim is the first entry in The Elder Scrolls to include Dragons in the game's 
wilderness. Like other creatures, Dragons are generated randomly in the world 
and will engage in combat. 
"""


sentences = parse_document(DOCUMENT)
norm_sentences = normalize_corpus(sentences,lemmatize=True) 
print("Total Sentences:", len(norm_sentences))

lsa_text_summarizer(norm_sentences, num_sentences=3,
                    num_topics=5, feature_type='frequency',
                    sv_threshold=0.5)  

textrank_text_summarizer(norm_sentences, num_sentences=3,
                         feature_type='tfidf')

# Compare the outputs of the two algorithms
# Also compare this with gensim's built-in algorithm based on TextRank
# What can you observe?
# Try this on a larger document collection and compare results

# SAQ: Try to use different summarization ratios and number of topics
