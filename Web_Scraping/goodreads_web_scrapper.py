#!/usr/bin/env python

from bs4 import *
from urllib.parse  import urljoin
import requests
import string
import os
import time
from multiprocessing import Pool
import re


def crawl(pages):
    indexed_url = [] # a list for sub-HTML websites in the main website
    for page in pages:
        if page not in indexed_url:
            r = requests.get(page)
            soup = BeautifulSoup(r.content, "html.parser")
            links = soup.find_all("a", {"class": "bookTitle"})
            for link in links:
                if 'href' in dict(link.attrs):
                    url = urljoin(page, link['href'])
                    if url.find("'") != -1:
                        continue
                    url = url.split('#')[0] 
                    if url[0:4] == 'http':
                        indexed_url.append(url)
    return indexed_url

def get_links():
    #                                      page number, query: a-z
    url = "https://www.goodreads.com/search?page={}&q={}&qid=EjxrVfJl1S&search_type=books&tab=books&utf8=%E2%9C%93"
    for page in range(1,2): # 1-100
        for i in range(1,len(string.ascii_lowercase)): # a-z
            pagelist.append(url.format(page,string.ascii_lowercase[i]))


def change_path():
    path = os.getcwd() + '/URLs'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    os.chdir(path)


def write_file(page):
    r = requests.get(page)
    soup = BeautifulSoup(r.content, "html.parser")
    title = soup.find("title").text.strip() 
    title = re.sub("[^0-9a-zA-Z]+", "_", title) + ".html"
    open(title, 'wb+').write(r.content)



if __name__ == "__main__":
    start = time.time()
    
    pagelist = []
    get_links()
    change_path()

    pool = Pool(24)
    pool.map(write_file,crawl(pagelist))
    pool.close()
    pool.join()

    end = time.time()
    print(end - start)