import sys
import os
from bs4 import *
import re
import json

BOOK_FILE = 'fuzzy_book_info.tsv'
REVIEW_FILE = 'fuzzy_review_info.tsv'

def write_into_file(file, string):
    file = open(file,'ab')
    file.write(string.encode('utf8'))
    file.close()

def extract_book_info(file):
    html = BeautifulSoup(file.read(), 'html.parser')
    if bool(html.find()) == False or html.body == None or html.head == None:
        return 
    
    url = html.head.find('link', attrs={'rel':'canonical'})['href']
    book_id = re.findall(r'\d+', url)[0]
    title = html.body.find('h1', attrs={'data-testid':'bookTitle'})
    if title == None:
        return
    title = title.text.strip()
    author = str(None)
    _json = html.head.find('script', attrs={'type':'application/ld+json'}).text
    if _json == None:
        return
    data = json.loads(_json)
    author = data["author"][0]["name"]
    data = data["aggregateRating"]
    if "ratingCount" in data:
        ratingCount = str(data["ratingCount"])
    else:
        ratingCount = str(None)
    if "ratingValue" in data:
        rating = str(data["ratingValue"])
    else:
        rating = str(None)
    if "reviewCount" in data:
        reviewCount = str(data["reviewCount"])
    else:
        reviewCount = str(None)


    year = html.body.find('p', attrs={'data-testid':'publicationInfo'})
    if year == None:
        year = str(None)
    else:
        year = str(re.findall(r'\d+', year.text)[-1])

    description = html.body.find('div', attrs={'data-testid':'description'})
    if description == None:
        description = str(None)
    else:
        description = description.find('span',attrs={'class': 'Formatted'}).text

    string = book_id + '\t' + title + '\t' + author
    string = string + '\t' + ratingCount + '\t' + reviewCount
    string = string + '\t' + rating + '\t' +  year + '\t' +  description + '\n'
    write_into_file(BOOK_FILE,string)
    extract_review_info(html, book_id, title)

def extract_review_info(html, book_id, title):

    reviews = html.body.findAll('section', attrs={'class':'ReviewText'})
    for i, _review in enumerate(reviews):
        review = _review.find('span',attrs={'class': 'Formatted'})
        for br in review.find_all('br'):
            br.replace_with(' ')
        review = review.text.replace('\n', ' ')
        if '(view spoiler)' in review:
            spoiler = str(True)
        else:
            spoiler = str(False)
        
        users = html.body.findAll('div', attrs={'class':'ReviewerProfile__name'})
        user = users[i].find('a')
        user_id = re.findall(r'\d+', user['href'])[0]
        user_name = user.text
        dates = html.body.findAll('span', {'class': 'Text Text__body3'})
        date = dates[i+1].find('a').text.replace('Edited','')
        ratings = html.body.findAll('section', attrs={'class':'ReviewCard__content'})
        if ratings[i].find('span', attrs={'class':'RatingStars RatingStars__small'}) == None:
            rating = str(None)
        else:
            rating = str(len(ratings[i].find_all('path', attrs={'class':'RatingStar__fill'})))

        string = book_id + '\t' + title + '\t' + user_id 
        string = string + '\t' + user_name + '\t' + date
        string = string + '\t' + rating + '\t' + spoiler
        string = string + '\t' + review + '\n'
        write_into_file(REVIEW_FILE,string)

if __name__ == '__main__':

    path = sys.argv[1]
    #if files exist then delete them
    if os.path.exists(BOOK_FILE):
        os.remove(BOOK_FILE)
    if os.path.exists(REVIEW_FILE):
        os.remove(REVIEW_FILE)
    
    for filename in os.listdir(os.path.abspath(path)):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
            try:
                extract_book_info(f)
            except:
                print("wrong file: ", f)
                continue