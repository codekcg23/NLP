# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

sentence = 'the quick brown fox jumped over the lazy dog'

# Since the Stanford parser is written in Java, we need to:
# 0. Install Java 1.8 or better
# 1. Download the Stanford Parser from https://nlp.stanford.edu/software/lex-parser.shtml
# 2. Unzip it to a directory and
# 3. Set environment variables to our JAVA path and this direcory
import os
java_path = r'/usr/bin/java'
os.environ['JAVAHOME'] = java_path

from nltk.parse.stanford import StanfordParser

scp = StanfordParser(model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')



result = list(scp.raw_parse(sentence))
print(result[0])

result[0].draw()

import nltk
from nltk.grammar import Nonterminal
from nltk.corpus import treebank

training_set = treebank.parsed_sents()

print(training_set[1])

# extract the productions for all annotated training sentences
treebank_productions = list(
                        set(production 
                            for sent in training_set  
                            for production in sent.productions()
                        )
                    )

treebank_productions[0:10]
  
# add productions for each word, POS tag
for word, tag in treebank.tagged_words():
	t = nltk.Tree.fromstring("("+ tag + " " + word  +")")
	for production in t.productions():
		treebank_productions.append(production)

# build the PCFG based grammar  
treebank_grammar = nltk.grammar.induce_pcfg(Nonterminal('S'), 
                                         treebank_productions)

# build the parser
viterbi_parser = nltk.ViterbiParser(treebank_grammar)

# get sample sentence tokens
tokens = nltk.word_tokenize(sentence)

# get parse tree for sample sentence
result = list(viterbi_parser.parse(tokens))


# get tokens and their POS tags from pattern package
from pattern.en import tag as pos_tagger
tagged_sent = pos_tagger(sentence)

# use NLTK POS tagger instead
tokens = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokens)

print(tagged_sent)

# extend productions for sample sentence tokens
for word, tag in tagged_sent:
    t = nltk.Tree.fromstring("("+ tag + " " + word  +")")
    for production in t.productions():
        treebank_productions.append(production)

# rebuild grammar
treebank_grammar = nltk.grammar.induce_pcfg(Nonterminal('S'), 
                                         treebank_productions)                                         

# rebuild parser
viterbi_parser = nltk.ViterbiParser(treebank_grammar)

# get parse tree for sample sentence
result = list(viterbi_parser.parse(tokens))

print(result[0])
result[0].draw()

                  