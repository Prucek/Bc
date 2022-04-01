import pandas as pd
from datasets import Dataset, DatasetDict
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import csv
from tqdm import tqdm
import os

# No interruptions
os.environ["WANDB_DISABLED"] = "true"

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


books.drop(books[books['genre'] == ''].index, inplace=True)
books.drop(books[books['genre'] == None].index, inplace=True)
books.drop(books[books['genre'].isin([])].index, inplace=True)
books.drop(books[books['summary'] == ''].index, inplace=True)
books.drop(books[books['summary'] == None].index, inplace=True)
books.drop(books[books['summary'].isin([])].index, inplace=True)

books.reset_index(drop=True, inplace=True)

genres = []
for i in books['genre']:
    genres.append(list(json.loads(i).values()))
books['genre'] = genres

all_genres = list(set(sum(genres, [])))
all_genres.sort()
genre_indexes = []
for genres in books['genre']:
    tmp = [all_genres.index(x) for x in genres]
    arr = []
    for i in range(len(all_genres)):
        arr.append(1 if i in tmp else 0)
    genre_indexes.append(arr)


books['genre_indexes'] = genre_indexes


"""
    For genre statistics
"""
counts = []
for i in range(len(all_genres)):
    counts.append(0)

for genre_idxs in books['genre']:
    tmp = [all_genres.index(x) for x in genre_idxs]
    for idx in tmp:
        counts[idx] = counts[idx] + 1

genre_count = {}
for idx, genre in enumerate(all_genres):
    genre_count[str(genre)] = counts[idx]

genre_count = {k: v for k, v in sorted(genre_count.items(), key=lambda item: item[1], reverse=True)}
# c = 0
# for i in genre_count:
#     if genre_count[str(i)] >= 10: 
#         print(i,genre_count[str(i)])
#         c = c + 1
# print(c)


# Removing ganres with less than 50 occurances
for idx, book_genres in enumerate(books['genre']):
    # print("out ", book_genres)
    for genre in book_genres:
        if genre_count[str(genre)] <= 50:
            # print("in ", books['genre'][idx])
            books['genre'][idx].remove(str(genre))

for idx, genres in enumerate(books['genre']):
    if genres == []:
        books.drop(idx,inplace=True) 


books.reset_index(drop=True, inplace=True)

all_genres = list(set(sum(books['genre'], [])))
all_genres.sort()

def clean_summary(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


books['summary'] = books['summary'].apply(lambda x: clean_summary(x))
books['summary'] = books['summary'].apply(lambda x: remove_stopwords(x))

dataset = pd.DataFrame(False, index=np.arange(len(data)),columns=all_genres)
dataset['book_name'] = books['book_name'].copy()
dataset['summary'] = books['summary'].copy()

dataset.drop(dataset[dataset['book_name'] == ''].index, inplace=True)
dataset.drop(dataset[dataset['book_name'] == None].index, inplace=True)
dataset.drop(dataset[dataset['book_name'].isna()].index, inplace=True)
dataset.drop(dataset[dataset['summary'] == ''].index, inplace=True)
dataset.drop(dataset[dataset['summary'] == None].index, inplace=True)
dataset.drop(dataset[dataset['summary'].isna()].index, inplace=True)


for i,genres in enumerate(books['genre']):
    for genre in genres:
        if str(genre) in dataset.columns:
            dataset.at[i, str(genre)] = True

# first and last words from summary
for i, summary in enumerate(books['summary']):
    # books['summary'][i] = summary[:128] + summary[len(summary)-382:] # head + tail
    tail = len(summary)-512
    if tail < 0 :
        tail = 0
    books['summary'][i] = summary[tail:] # just tail

dset = Dataset.from_pandas(dataset)

dset = dset.remove_columns([ '__index_level_0__'])
dset = dset.filter(lambda example: example['summary'] is not None)

# for i in range(1,5):
#     # print("test ",[id2label[idx] for idx, label in enumerate(reloaded_split_dataset['test'][i]) if label == True])
#     # print("train ",[id2label[idx] for idx, label in enumerate(reloaded_split_dataset['train'][i]) if label == True])
#     arr = []
#     for label in all_genres:
#         if dset[str(label)][-i] == True: 
#             arr.append(label)
#     print(dset['book_name'][-i])
#     print(arr)

# exit()
# split_dataset = dset.train_test_split(test_size=0.1, shuffle=True)

# 90% train, 10% test + validation
train_test_valid = dset.train_test_split(test_size=0.1, shuffle=True)
# Split the 10% test + valid in half test, half valid
test_valid = train_test_valid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_test_valid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})


train_test_valid_dataset.save_to_disk("./processed_booksumaries_50+")