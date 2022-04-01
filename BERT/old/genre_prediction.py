import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

data = []

with open("booksummaries.txt", 'r', encoding="utf8") as f:
    reader = csv.reader(f, dialect='excel-tab', delimiter="\t")
    for row in tqdm(reader):
        data.append(row)

book_id = []
book_name = []
summary = []
genre = []

for i in tqdm(data):
    book_id.append(i[0])
    book_name.append(i[2])
    genre.append(i[5])
    summary.append(i[6])

books = pd.DataFrame({'book_id': book_id, 'book_name': book_name,
                       'genre': genre, 'summary': summary})

books.drop(books[books['genre'] ==''].index, inplace=True)

genres = []
for i in books['genre']:
    genres.append(list(json.loads(i).values()))
books['genre'] = genres

all_genres = list(set(sum(genres, [])))

# def clean_summary(text):
#     text = re.sub("\'", "", text)
#     text = re.sub("[^a-zA-Z]"," ",text)
#     text = ' '.join(text.split())
#     text = text.lower()
#     return text
#
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
#
# def remove_stopwords(text):
#     no_stopword_text = [w for w in text.split() if not w in stop_words]
#     return ' '.join(no_stopword_text)
#
# books['summary'] = books['summary'].apply(lambda x: clean_summary(x))
# books['summary'] = books['summary'].apply(lambda x: remove_stopwords(x))


sequence_to_classify = books['summary'][0]


from transformers import pipeline
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers")
retval = classifier(sequence_to_classify, all_genres, multi_label=True)


for idx, score in enumerate(retval['scores']):
    if score > 0.5:
        print(retval['labels'][idx])

