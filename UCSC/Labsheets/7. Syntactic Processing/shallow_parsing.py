# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

# Training your own chunker using chunked treeband data - again made available in NLTK!
# As before (with tagging) we first divide the data into training and testing sets
from nltk.corpus import treebank_chunk
data = treebank_chunk.chunked_sents()
train_data = data[0:3500]
test_data = data[3500:]
print(train_data[0])

simple_sentence = 'the brown fox jumped over the lazy dog'

# Can use tagger from package pattern.en if using Python 2.x
# from pattern.en import tag
# tagged_sentence = tag(sentence)

import nltk
from nltk.chunk import RegexpParser
tokens = nltk.word_tokenize(simple_sentence)
tagged_simple_sent = nltk.pos_tag(tokens)
print(tagged_simple_sent)

# We first define our grammars using regex pattern using the RegexpParser
# We can specify which patterns we want to segment in a sentence as *chunks*
chunk_grammar = """
NP: {<DT>?<JJ>*<NN.*>}
VP: {<VBD><IN>}
"""
rc = RegexpParser(chunk_grammar)
c = rc.parse(tagged_simple_sent)
print(c)

# We sometimes want to specify which patterns we DO NOT want to segment in a sentence
# so that we can *chunk* all the others
chink_grammar = """
NP: {<JJ|NN>+} # chunk only adjective-noun pair as NP
"""

rc = RegexpParser(chink_grammar)
c = rc.parse(tagged_simple_sent)
print(c)


# A more realistic grammar for chunking
grammar = """
NP: {<DT>?<JJ>?<NN.*>}  
ADJP: {<JJ>}
ADVP: {<RB.*>}
PP: {<IN>}      
VP: {<MD>?<VB.*>+}

"""

# And a more realistic sentence as input
sentence = 'the brown fox is quick and he may jump over the lazy dog'
tokens = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokens)

rc = RegexpParser(grammar)
c = rc.parse(tagged_sent)
print(c)

print(rc.evaluate(test_data))
# The performance is not great!
# Why is this?


# We have acecss to a utility function tree2conlltags which extracts word, tag and
# chunk triples from annotated text
from nltk.chunk.util import tree2conlltags, conlltags2tree
# Let's take a slightly more typical sentence from our data
train_sent = train_data[7]
print(train_sent)

# We extract the POS and chunk tags using tree2conlltags function which returns a list of tuples
wtc = tree2conlltags(train_sent)
wtc

# We can 'reverse' this to output a shallow tree using the conlltags2tree function
tree = conlltags2tree(wtc)
print(tree)


# We can use these features to train a 'combined' chunker as we did for POS tagging
def conll_tag_chunks(chunk_sents):
  tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
  return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]
  
def combined_tagger(train_data, taggers, backoff=None):
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff
  
from nltk.tag import UnigramTagger, BigramTagger
from nltk.chunk import ChunkParserI

# We create a new class to use the word, POS and Chunk tag features to train a chunker
# that is able to 'backoff' from bigram to a unigram model as before
# Can you have another layer for trigram and back off to this model?
class NGramTagChunker(ChunkParserI):
    
  def __init__(self, train_sentences, 
               tagger_classes=[UnigramTagger, BigramTagger]):
    train_sent_tags = conll_tag_chunks(train_sentences)
    self.chunk_tagger = combined_tagger(train_sent_tags, tagger_classes)

  def parse(self, tagged_sentence):
    if not tagged_sentence: 
        return None
    pos_tags = [tag for word, tag in tagged_sentence]
    chunk_pos_tags = self.chunk_tagger.tag(pos_tags)
    chunk_tags = [chunk_tag for (pos_tag, chunk_tag) in chunk_pos_tags]
    wpc_tags = [(word, pos_tag, chunk_tag) for ((word, pos_tag), chunk_tag)
                     in zip(tagged_sentence, chunk_tags)]
    return conlltags2tree(wpc_tags)

# We call our new class and pass it the training data from the chunked treebank
ntc = NGramTagChunker(train_data)
print(ntc.evaluate(test_data))
# Now we get really good results on the data set
# Why?

# Let's try to visualize the chunk for the more realistic sample sentence
tree = ntc.parse(tagged_sent)
print(tree)
tree.draw()


# We now use our shallow parser on a larger 'Wall Street Journal' corpus
# SAQ 1. How big is it?
# SAQ 2. How much test data did we have before, and how much now?
from nltk.corpus import conll2000
wsj_data = conll2000.chunked_sents()
train_wsj_data = wsj_data[:7500]
test_wsj_data = wsj_data[7500:]
print(train_wsj_data[10])

# We first train our model on the training corpus
tc = NGramTagChunker(train_wsj_data)
# And then we test it on the test data
print(tc.evaluate(test_wsj_data))
