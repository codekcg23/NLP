# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""


sentence = 'Colorless green ideas sleep furiously'

from spacy.en import English
parser = English()
parsed_sent = parser(sentence)

dependency_pattern = '{left}<---{word}[{w_type}]--->{right}\n--------'
for token in parsed_sent:
    print(dependency_pattern.format(word=token.orth_, 
                                  w_type=token.dep_,
                                  left=[t.orth_ 
                                            for t 
                                            in token.lefts],
                                  right=[t.orth_ 
                                             for t 
                                             in token.rights]))
                                             

# Since the Stanford Dependency parser is written in Java, we need to:
# 0. Install Java 1.8 or better
# 1. Download the Stanford Parser from https://nlp.stanford.edu/software/lex-parser.shtml
# 2. Unzip it to a directory and
# 3. Set environment variables to our JAVA path and this direcory
import os
java_path = r'/usr/bin/java'
os.environ['JAVAHOME'] = java_path
                                             
from nltk.parse.stanford import StanfordDependencyParser

"""
Previous version of NLTK/Stanford Parser syntax
sdp = StanfordDependencyParser(path_to_jar='/Users/arw/Documents/Work/stanford-parser-full/stanford-parser.jar',
                               model_path='/Users/arw/Documents/Work/stanford-parser-full/stanford-parser-3.8.0-models.jar')
"""

sdp = StanfordDependencyParser(model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')


result = list(sdp.raw_parse(sentence))  

"""
result[0]

[item for item in result[0]]
"""

dep_tree = [parse for parse in result][0]
print(dep_tree)
dep_tree

# generation of annotated dependency tree
from graphviz import Source
dep_tree_dot_repr = [parse for parse in result][0].to_dot()
source = Source(dep_tree_dot_repr, filename="dep_tree", format="png")
source.view()
             
import nltk
tokens = nltk.word_tokenize(sentence)

dependency_rules = """
'fox' -> 'The' | 'brown'
'quick' -> 'fox' | 'is' | 'and' | 'jumping'
'jumping' -> 'he' | 'is' | 'dog'
'dog' -> 'over' | 'the' | 'lazy'
"""

dependency_grammar = nltk.grammar.DependencyGrammar.fromstring(dependency_rules)
print(dependency_grammar)

dp = nltk.ProjectiveDependencyParser(dependency_grammar)
res = [item for item in dp.parse(tokens)]
tree = res[0] 
print(tree)

tree.draw()                          
