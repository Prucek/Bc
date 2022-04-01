#!/usr/bin/env python

import sys
import os
from bs4 import *
import json
import socket
import re


def get_review_count_from_fuzzy_files(html):
    try:
        _json = html.head.find('script', attrs={'type':'application/ld+json'}).text
        data = json.loads(_json)
        return int(data["aggregateRating"]["reviewCount"])
    except:
        return None


def write_into_file(string):
    host = str(socket.gethostname())
    last_path = str(os.path.basename(os.path.normpath(sys.argv[1])))
    filename = "statistic." + host + "." + last_path + ".tsv"
    file = open(filename,'ab')
    file.write(string.encode('utf8'))
    file.close()


def extract_book_info(file, filename):

    html = BeautifulSoup(file.read(), 'html.parser')

    #is not html file
    if bool(html.find()) == False or html.body == None:
        return None

    is_ok = True
    link = html.head.find('link', attrs={'rel':'canonical'})
    title = html.body.find('h1', attrs={'id':'bookTitle'})
    reviewCount = None
    if title == None:
        is_ok = False
        reviewCount = get_review_count_from_fuzzy_files(html)
    else:
        reviewCount = html.body.find('meta', attrs={'itemprop':'reviewCount'})['content']
    
    if link == None:
        return None
    
    num = re.findall(r'\d+', filename)[0]

    string = num + '\t' + link['href'] + '\t' + str(reviewCount) + '\t' + str(is_ok) + '\n'
    write_into_file(string)

    if is_ok:
        return True
    else:
        return False


if __name__ == '__main__':

    path = sys.argv[1]
    
    ok = 0
    fuzzy = 0
    wrong = 0
    for subdir, dirs, files in os.walk(path):
        try:
            for file in files:
                with open(os.path.join(subdir, file), 'r', encoding='utf-8') as f:
                    value = extract_book_info(f, file)
                    if value == True:
                        ok = ok + 1
                    elif value == False:
                        fuzzy = fuzzy + 1
                    elif value == None:
                        wrong = wrong + 1
        except:
            pass

    print(ok, fuzzy, wrong)
