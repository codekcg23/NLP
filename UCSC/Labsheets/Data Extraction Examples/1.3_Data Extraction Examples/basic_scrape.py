#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:34:54 2017

@author: arw
"""

import requests   # This is a package to send requests for web pages

# We will be trying to extract data from a prison site
url = 'http://www.showmeboone.com/sheriff/JailResidents/JailResidents.asp'

response = requests.get(url)
html = response.content # We are only interested in the main content

# First just let's print what we get!
print(html)

# Now, we're going to parse it (probably better to comment the print statement above)

from bs4 import BeautifulSoup # This is a package for extracting markup

soup = BeautifulSoup(html)
print(soup.prettify())  # BeautifulSoup pretty prints the HTML!

# At this point we need to inspect the 'source' of our html page to see how it is formatted
table = soup.find('tbody', attrs={'class': 'stripe'})

print(table.prettify()) # Pretty print the entire table

# Pretty print row by row
for row in table.findAll('tr'):
    print(row.prettify())

# Pretty print cell by cell
for row in table.findAll('tr'):
    for cell in row.findAll('td'):
        print(cell.text)
        
# Remove the annoying non-break-space      
for row in table.findAll('tr'):
    for cell in row.findAll('td'):
        print(cell.text.replace('&nbsp;', ''))

# Print as a Python List       
for row in table.findAll('tr'):
    list_of_cells = []
    for cell in row.findAll('td'):
        text = cell.text.replace('&nbsp;', '')
        list_of_cells.append(text)
    print(list_of_cells)  
    
# Make it into a list of lists
list_of_rows = []
for row in table.findAll('tr'):
    list_of_cells = []
    for cell in row.findAll('td'):
        text = cell.text.replace('&nbsp;', '')
        list_of_cells.append(text)
    list_of_rows.append(list_of_cells)

print(list_of_rows)


# Finally we may want to write it as a CSV file and view using spreadsheet
import csv

list_of_rows = []
for row in table.findAll('tr'):
    list_of_cells = []
    for cell in row.findAll('td'):
        text = cell.text.replace('&nbsp;', '')
        list_of_cells.append(text)
    list_of_rows.append(list_of_cells)

outfile = open("./inmates.csv", "w")
writer = csv.writer(outfile)
writer.writerows(list_of_rows)
outfile.close()






