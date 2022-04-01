#!/usr/bin/env python
# from Goodreads
# Extracting basic information about a book into a .tsv 
# Extracting reviews of a book into a .tsv
# Creates book_info.tsv from all files in given direcotry
# Creates review_info.tsv from all books in given direcotry
# Creates more_reviews.txt from books which contains URL-s with additional reviews
# Creates to_download_via_selenium.txt - URL-s to be downloaded again via selenium
# Author: Peter Rucek
# Date: 28.10.2021
# Usage: python3 extract_book_info.py <path_to_directory_where_downloaded_html_are_located>

import sys
import os
from bs4 import *
import re
import json

BOOK_FILE = 'book_info.tsv'
REVIEW_FILE = 'review_info.tsv'
MORE_REVIEWS = 'more_reviews.txt'
#TO_DOWNLOAD = 'to_download_via_selenium.txt'

# to_donload_again_count = 0


# def to_donload_again(filename, url, review_count):
#     global to_donload_again_count

#     pattern = re.compile("^[0-9]+$")
#     match = pattern.fullmatch(filename)

#     if match == None:
#         if os.name == "nt": 
#             params = filename.split('%3F')[1]
#         else:
#             params = filename.split('?')[1]
        
#         write_into_file(TO_DOWNLOAD, url + "?" + params + "\n")

#     elif filename == match.group(0):
#         count = 1
#         if not review_count: # Even more "broken" file
#             write_into_file(TO_DOWNLOAD,url + "\n")
#             return

#         if review_count > 30 and review_count  < 300:
#             count = (review_count // 30 ) + 1
#         elif  review_count > 300:
#             count = 10

#         for i in range(1, count + 1):
#             to_download = url + '?csm_scope=&amp;hide_last_page=true&amp;language_code=en&amp;page=' + str(i) + '\n'
#             write_into_file(TO_DOWNLOAD,to_download)

#     else:
#         return

#     to_donload_again_count += 1


def get_review_count_from_fuzzy_files(html):
    try:
        _json = html.head.find('script', attrs={'type':'application/ld+json'}).text
        data = json.loads(_json)
        return int(data["aggregateRating"]["reviewCount"])
    except:
        return None


def check_file_get_title(html, filename):
    #is not html file
    if bool(html.find()) == False or html.body == None:
        return None

    title = html.body.find('h1', attrs={'id':'bookTitle'})
    if title == None:
        link = html.head.find('link', attrs={'rel':'canonical'})
        if link == None :
            return None

        #to_donload_again(filename,link['href'], get_review_count_from_fuzzy_files(html))   
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
    file = open(file,'ab')
    file.write(string.encode('utf8'))
    file.close()


def extract_book_info(file, filename):

    parsed_html = BeautifulSoup(file.read(), 'html.parser')

    title = check_file_get_title(parsed_html, filename) 
    if title == None:
        return False

    title = title.text.strip()
    bookID = get_bookID(parsed_html)
    author = parsed_html.body.find('span', attrs={'itemprop':'name'})
    if author:
        author = author.text.strip()
    else:
        author = str(None)
    ratingCount = parsed_html.body.find('meta', attrs={'itemprop':'ratingCount'})['content']
    reviewCount = parsed_html.body.find('meta', attrs={'itemprop':'reviewCount'})['content']
    rating = parsed_html.body.find('span', attrs={'itemprop':'ratingValue'})
    if rating:
        rating = rating.text.strip()
    else:
        rating = str(None)
    year = get_year(parsed_html)
    description = get_description(parsed_html)

    # write to file
    string = bookID + '\t' + title + '\t' + author
    string = string + '\t' + ratingCount + '\t' + reviewCount
    string = string + '\t' + rating + '\t' +  year + '\t' +  description + '\n'
    write_into_file(BOOK_FILE,string)

    value = extract_review_info(parsed_html, filename)
    print('Written ',value,' reviews of ', title,' into ', REVIEW_FILE)


    link = parsed_html.head.find('link', attrs={'rel':'canonical'})['href']
    select =  parsed_html.body.find('select', attrs={'id':'language_code'})
    englishReviewsCount = select.find('option',attrs={'value':'en'})
    if englishReviewsCount == None :
        return True

    num = re.findall(r'(\d+)', englishReviewsCount.text)
    englishReviewsCount = int(num[0])
    count = 1
    if englishReviewsCount > 30 and englishReviewsCount  < 300:
        count = (englishReviewsCount // 30 ) + 1
    elif  englishReviewsCount > 300:
        count = 10

    for i in range(2, count + 1):
        moreReviews = link + '?csm_scope=&amp;hide_last_page=true&amp;language_code=en&amp;page=' + str(i) + '\n'
        write_into_file(MORE_REVIEWS,moreReviews)

    return True


def extract_review_info(html, filename):

    title = check_file_get_title(html, filename) 
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
        if rating == str(0):
            continue
        freeText = a.find_all('div', class_= 'reviewText')
        spoiler = str(None)
        review = str(None)
        if freeText != []:
            reviewText = freeText[0].find_all('span', {'id': re.compile(r'freeText.')})
            if len(reviewText) > 0:
                review = reviewText[len(reviewText) - 1]
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
    if os.path.exists(MORE_REVIEWS):
        os.remove(MORE_REVIEWS)
    # if os.path.exists(TO_DOWNLOAD):
    #     os.remove(TO_DOWNLOAD)
    
    #try:
    ok = 0
    wrong = 0
    for filename in os.listdir(os.path.abspath(path)):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
            value = extract_book_info(f, filename)
            if value == True:
                ok = ok + 1
            if value == False:
                wrong = wrong + 1
    #except:
    #    print('Wrong directories or files')

    print('Written ',ok,' book infos into ', BOOK_FILE)
    print('There were ', wrong , ' wrong files in this direcotry')
    #print(to_donload_again_count, ' files have to be downloaded again')