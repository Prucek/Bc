import platform
import socket
import sys
import os
from bs4 import *
import re


def check_file_get_title(html, filename):
    #is not html file
    if bool(html.find()) == False or html.body == None:
        return None
    
    is_keywords_file = re.search("^.*\.keywords\.html$", filename)
    if not is_keywords_file:
        return None

    titles = html.body.findAll('a', attrs={'itemprop':'url'})
    if titles[0] == None:
        return None
    
    string = str()
    for title in titles:
        string = string + title.text.strip()
        if titles[-1] != title:
            string = string +  " - " 

    return string

def get_ID(html):
    url = html.head.find('link', attrs={'rel':'canonical'})['href']
    return re.findall(r'\d+', url)[0]

def write_into_file(string):
    host = str(socket.gethostname())
    last_path = str(os.path.basename(os.path.normpath(sys.argv[1])))
    filename = 'plot_keywords.' + host + "." + last_path + ".tsv"
    file = open(filename,'ab')
    file.write(string.encode('utf8'))
    file.close()

def check_if_has_plotsummary(id, dir):
    if platform.system() == 'Linux':
        string = str(dir) + '/tt' + str(id) + '.plotsummary.html'
    if platform.system() == 'Windows':
        string = str(dir) + '\\tt' + str(id) + '.plotsummary.html'
    try:
        file = open(string, 'r', encoding='utf-8')
    except:
        file = None
    finally:
        return file


def extract_keywords(file, filename, dir):
    parsed_html = BeautifulSoup(file.read(), 'html.parser')

    title = check_file_get_title(parsed_html, filename) 
    if title == None:
        return

    id = get_ID(parsed_html)

    plotsummary = check_if_has_plotsummary(id, dir)
    if plotsummary == None:
        return
        
    plotsummary_html = BeautifulSoup(plotsummary.read(), 'html.parser')
    summaries = plotsummary_html.body.findAll('li', attrs={'class':'ipl-zebra-list__item'})
    sumary_synopsis = []
    for summary in summaries:
        if summary['id'] == 'no-synopsis-content' :
            summary.decompose()
            continue
        _summary = summary.find('p')
        if _summary == None:
            _summary = summary
        sumary_synopsis.append(_summary.text.rstrip().replace("\n", ""))
    
    divs = parsed_html.body.findAll('div', attrs={'class':'sodatext'})
    keywords = []
    for div in divs:
        tmp = div.find('a').text
        keywords.append(tmp)
    
    string = id + '\t' + title + '\t' 
    for keyword in keywords:
        string = string + keyword + '~'
    string = string + '\t'
    for s in sumary_synopsis:
        string = string + s + '~'
    string = string + '\n'

    write_into_file(string)


if __name__ == '__main__':

    path = sys.argv[1]
    for subdir, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(subdir, file), 'r', encoding='utf-8') as f:
                extract_keywords(f,file, subdir)