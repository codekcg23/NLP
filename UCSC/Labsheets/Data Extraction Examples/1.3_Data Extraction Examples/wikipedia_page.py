#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 12:52:55 2017

@author: arw
"""

from bs4 import BeautifulSoup
import requests

url ='https://si.wikipedia.org/wiki/‍2021_යුරෝපියානු_ජලගැලුම්'
# get contents from url
content = requests.get(url).content
# get soup
soup = BeautifulSoup(content,'lxml') # choose lxml parser
# find the tag : <div class="toc">
tag = soup.find('div', {'class' : 'toc'}) # id="toc" also works

"""
# get all the links
links = tag.findAll('a') # <a href='/path/to/div'>topic</a>
# print them 
for link in links:
  print(link.text) # get text from <a>

# find the tag : <img ... >
image_tags = soup.findAll('img')
# print out image urls
for image_tag in image_tags:
    print(image_tag.get('src'))


# find all the references
ref_tags = soup.findAll('span', { 'class' : 'reference-text' })
# iterate through the ResultSet
for i,ref_tag in enumerate(ref_tags):
    # print text only
    print(u'[{0}] {1}'.format(i,ref_tag.text))

"""
# find all the paragraph tags
p_tags = soup.findAll('p')

# gather all <a> tags 
a_tags = []
for p_tag in p_tags:
    a_tags.extend(p_tag.findAll('a'))
# filter the list : remove invalid links
a_tags = [ a_tag for a_tag in a_tags if 'title' in a_tag.attrs and 'href' in a_tag.attrs ]
# print all links
f = open('content.txt', encoding='utf-8', mode='w')
for i,a_tag in enumerate(a_tags):
    f.write(u'[{0}] {1} -> {2}'.format(i,a_tag.get('title'),a_tag.get('href'))+ '\n')

#    print(u'[{0}] {1} -> {2}'.format(i,a_tag.get('title'),a_tag.get('href')))

