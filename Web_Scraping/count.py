import sys
import os
from bs4 import *

path = sys.argv[1]
wrong = 0
fuzzy = 0
for filename in os.listdir(os.path.abspath(path)):
    with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
        html = BeautifulSoup(f.read(), 'html.parser')
        if bool(html.find()) == False or html.body == None or html.head == None:
            wrong = wrong + 1
            continue
        title = html.body.find('h1', attrs={'data-testid':'bookTitle'})
        if title == None:
            wrong = wrong + 1
            continue
        fuzzy = fuzzy + 1

print("Wrong: ", wrong)
print("Fuzzy: ", fuzzy)

