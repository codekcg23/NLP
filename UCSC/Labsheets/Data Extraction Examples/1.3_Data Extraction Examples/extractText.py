#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 07:23:02 2017

@author: arw
"""

import re
import urllib.request
import requests
from bs4 import BeautifulSoup as bs
 
# html = urllib.request.urlopen('https://vinothramachandra.wordpress.com')

url='https://ta.wikipedia.org/wiki/கொழும்பு'
html = requests.get(url).content


soup = bs(html, 'html.parser')

# Can use alternate parsers lxml or html5lib instead of html.parser

data = soup.findAll(text=True)
 
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title', 'meta', 'Start', 'Post', 'div']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True
 
result = filter(visible, data)

# Make a string out of the list
result = ''.join(result)

# Remove common escape sequences
result = re.sub(r'[\t\r\n]', '', result)
 

print(result)


