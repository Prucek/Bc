#!/usr/bin/env python
# from Goodreads
# Extracting basic information about a book into a .tsv 
# Extracting reviews of a book into a .tsv
# Creates book_info.tsv from all files in given direcotry
# Creates review_info.tsv from all books in given direcotry
# Author: Peter Rucek
# Date: 28.10.2021
# Usage: python3 extract_book_info.py <path_to_directory_where_downloaded_html_are_located>

import sys
import os
from bs4 import *
import re

BOOK_FILE = 'book_info.tsv'
REVIEW_FILE = 'review_info.tsv'


def check_file_get_title(html):
    #is not html file
    if bool(html.find()) == False or html.body == None:
        return None

    title = html.body.find('h1', attrs={'id':'bookTitle'})
    if title == None:
        #skip file
        return None
    return title


def get_bookID(html):
    url = html.head.find('link', attrs={'rel':'canonical'})['href']
    return re.findall(r'\d+', url)[0]


def get_year(html):
    # extract date of publication 
    published = html.body.find('div', attrs={'id':'details'})
    published = published.find_all('div', attrs={'class':'row'})
    year = str(0)
    if published:
        published = published[len(published) - 1].text
        tmp = ' '.join(published.split())
        year = re.findall(r'(\d{4})',published)
        if year:
            year = year[len(year) - 1]
        else:
            year = str(0)
    return year

def get_description(html):
    #extract long description
    description = html.body.find('div', attrs={'id':'description'})
    if description:
        description = description.find_all('span')
        description = description[len(description) - 1].text
        tmp = ' '.join(description.split())
        description = tmp 
    else: 
        description = str(0)
    return description

def write_into_file(file, string):
    file= open(file,'ab')
    file.write(string.encode('utf8'))
    file.close()


def extract_book_info(file):

    parsed_html = BeautifulSoup(file.read(), 'html.parser')

    title = check_file_get_title(parsed_html) 
    if title == None:
        return False

    title = title.text.strip()
    bookID = get_bookID(parsed_html)
    author = parsed_html.body.find('span', attrs={'itemprop':'name'}).text.strip()
    ratingCount = parsed_html.body.find('meta', attrs={'itemprop':'ratingCount'})['content']
    reviewCount = parsed_html.body.find('meta', attrs={'itemprop':'reviewCount'})['content']
    rating = parsed_html.body.find('span', attrs={'itemprop':'ratingValue'}).text.strip()
    year = get_year(parsed_html)
    description = get_description(parsed_html)

    # write to file
    string = bookID + '\t' + title + '\t' + author
    string = string + '\t' + ratingCount + '\t' + reviewCount
    string = string + '\t' + rating + '\t' +  year + '\t' +  description + '\n'
    write_into_file(BOOK_FILE,string)

    value = extract_review_info(parsed_html)
    print('Written ',value,' reviews of ', title,' into ', REVIEW_FILE)

    return True


def extract_review_info(html):

    title = check_file_get_title(html) 
    title = title.text.strip()
    bookID = get_bookID(html)

    reviews = html.body.find('div', attrs={'id':'bookReviews'})
    if reviews.find_all('div', attrs={'class':'nocontent'}) != []:
        return 0


    count = 0
    arr = reviews.find_all('div',  attrs={'class':'review'})
    for a in arr:
        tag = a.find_all('a', class_='imgcol')[0]
        userName = tag['title']
        userID = re.findall(r'\d+', tag['href'])[0]
        date = a.find_all('a', class_='reviewDate')[0].text
        rating = str(len(a.find_all('span', class_='staticStar p10'))) # out of 5 rating (0 means no rating added by user)
        freeText = a.find_all('div', class_= 'reviewText')[0].find_all('span', {'id': re.compile(r'freeText.')})
        
        review = freeText[len(freeText) - 1]
        for br in review.find_all('br'):
            br.replace_with(' ')

        review = review.text.replace('\n', ' ')

        if '(view spoiler)' in review:
            spoiler = str(True)
        else:
            spoiler = str(False)

        count = count + 1
        string = bookID + '\t' + title + '\t' + userID 
        string = string + '\t' + userName + '\t' + date
        string = string + '\t' + rating + '\t' + spoiler
        string = string + '\t' + review + '\n'

        write_into_file(REVIEW_FILE,string)


    return count

if __name__ == '__main__':

    path = sys.argv[1]
    #if files exist the delete them
    if os.path.exists(BOOK_FILE):
        os.remove(BOOK_FILE)
    if os.path.exists(REVIEW_FILE):
        os.remove(REVIEW_FILE)
    # try:
    ok = 0
    wrong = 0
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
            value = extract_book_info(f)
            if value == True:
                ok = ok + 1
            if value == False:
                wrong = wrong + 1
    # except:
    #     print('Wrong directories or files')

    print('Written ',ok,' book infos into ', BOOK_FILE)
    print('There were ', wrong, ' wrong files in this direcotry')