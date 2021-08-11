# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

sentence = "I saw the man with the telescope but he didn't see me"


# Using NLTK's built-in tagger based on PTB
import nltk
tokens = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokens, tagset='universal')
print(tagged_sent)

    
# Using the pattern package (Python 2.x only) built-in tagger (optional)
from pattern.en import tag
tagged_sent = tag(sentence)
print(tagged_sent)



# Building your own tagger
# - default tagger that tags all words the same!
# - regex tagger that doesn't care about context (most common tag per word)

# Fortunately the treebank corpus is bundled with NLTK for training a tagger
# We need to divide the data into training and test sets first
from nltk.corpus import treebank
data = treebank.tagged_sents()
train_data = data[:3500]
test_data = data[3500:]
print(train_data[0])

# SAQ 1: How much data is there for training, testing?
# SAQ 2: What is the last training sentence; test sentence?

# Default 'naive' tagger - tags all words with a given tag!
from nltk.tag import DefaultTagger
dt = DefaultTagger('NN') # Can specify any default tag - NN gives best score - why?

# Test score and example sentence tag output
print(dt.evaluate(test_data))
print(dt.tag(tokens))


# Regex tagger
from nltk.tag import RegexpTagger
# Define 'fixed' regex tag patterns
patterns = [
        (r'.*ing$', 'VBG'),               # gerunds
        (r'.*ed$', 'VBD'),                # simple past
        (r'.*es$', 'VBZ'),                # 3rd singular present
        (r'.*ould$', 'MD'),               # modals
        (r'.*\'s$', 'NN$'),               # possessive nouns
        (r'.*s$', 'NNS'),                 # plural nouns
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'.*', 'NN')                     # nouns (default) ... 
]
rt = RegexpTagger(patterns)

# Test score and example sentence tag output
print(rt.evaluate(test_data))
print(rt.tag(tokens))


# Training your own tagger
# 1. using n-gram taggers and combining them with backoff
# 2. using naive bayes (statistical) model
# 3. using maximum entropy (classifier) model

## N gram taggers
from nltk.tag import UnigramTagger # Context insentitive
from nltk.tag import BigramTagger  # Considers previous word
from nltk.tag import TrigramTagger # Considers previous 2 words

# Traing the taggers
ut = UnigramTagger(train_data)
bt = BigramTagger(train_data)
tt = TrigramTagger(train_data)

# Test UnigramTagger score and example sentence tag output
print(ut.evaluate(test_data))
print(ut.tag(tokens))

# Test BigramTagger score and example sentence tag output
print(bt.evaluate(test_data))
print(bt.tag(tokens))

# Test TrigramTagger score and example sentence tag output
print(tt.evaluate(test_data))
print(tt.tag(tokens))

# Combining all 3 n-gram taggers with backoff (smoothing)
def combined_tagger(train_data, taggers, backoff=None):
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff

ct = combined_tagger(train_data=train_data, 
                     taggers=[UnigramTagger, BigramTagger, TrigramTagger],
                     backoff=rt)

# Test Combined n-gram tagger score and example sentence tag output
print(ct.evaluate(test_data))       
print(ct.tag(tokens))



# Treating POS tagging as a classification problem
# We use the ClassifierBasedPOSTagger class to build a classifier by specifying some
# classification algorithm - here the NaiveBayes abd Maxent algorithms which are passed
# to the class via the classifier_builder parameter
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.tag.sequential import ClassifierBasedPOSTagger

# First a Naive Bayes (statistical) classifier
nbt = ClassifierBasedPOSTagger(train=train_data,
                               classifier_builder=NaiveBayesClassifier.train)

# Test NBC tagger score and example sentence tag output
print(nbt.evaluate(test_data))
print(nbt.tag(tokens))    


# Finally a Maximum entropy classifier (that would take sometime)
# met = ClassifierBasedPOSTagger(train=train_data,
#                               classifier_builder=MaxentClassifier.train)

met = ClassifierBasedPOSTagger(train=train_data, 
                               classifier_builder=lambda train_feats: MaxentClassifier.train(train_feats, max_iter=10))

# Test Maxent tagger score and example sentence tag output
print(met.evaluate(test_data))                           
print(met.tag(tokens))


# Final accuracies
print('Tagger accuracies:')
print()
print('Default tagger %.2f' %dt.evaluate(test_data))
print('Regex tagger %.2f' %rt.evaluate(test_data))
print('Unigram tagger %.2f' %ut.evaluate(test_data))
print('Bigram tagger %.2f' %bt.evaluate(test_data))
print('Trigram tagger %.2f' %tt.evaluate(test_data))
print('Combined tagger %.2f' %ct.evaluate(test_data))
print('Naive Bayes tagger %.2f' %nbt.evaluate(test_data))
print('Maxent tagger %.2f' %met.evaluate(test_data))
