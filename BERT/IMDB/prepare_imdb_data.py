from io import StringIO
from operator import indexOf
import platform
import pandas as pd
from datasets import Dataset, DatasetDict
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import csv
import sys
from tqdm import tqdm
import os

BERT_LEN = 512

nltk.download('stopwords')
plot_keywords = []
global_movies = pd.DataFrame({'id': [], 'name': [],
                       'keywords': [], 'summary': []})

if platform.system() == 'Linux':
    file = "./plot_keywords.txt"
elif platform.system() == 'Windows':
    file =".\plot_keywords.txt"

with open(file, 'r', encoding="utf8") as f:
    for row in f:
        row = ' '.join(row.split())
        row = row.lower()
        plot_keywords.append(row)


def clean_summary(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


def parse(f, filename):
    is_keywords_file = re.search("^plot_keywords.*.tsv$", filename)
    if not is_keywords_file:
        return None
    movie_data = []
    reader = csv.reader(f, dialect='excel-tab', delimiter="\t")
    # count = 0
    for row in tqdm(reader):
        movie_data.append(row)
        # count = count + 1
        # if count > 10:
        #     break
      
    id = []
    name = []
    keywords = []
    summary = []

    for i in tqdm(movie_data):
        
        if i[2] == None or i[2] == "":
            continue
        reader = csv.reader(StringIO(i[3]), delimiter="~")
        _summary = []
        for row in reader:
            _summary.append(row)
        _summary[0].pop(-1)

        for j in range(len(_summary[0])):
            id.append(i[0])
            name.append(i[1])
            
            reader = csv.reader(StringIO(i[2]), delimiter="~")
            _keywords = []
            for row in reader:
                _keywords.append(row)
            _keywords[0].pop(-1)
            keywords.append(_keywords[0])
            summary.append(_summary[0][j])


    movies = pd.DataFrame({'id': id, 'name': name,
                       'keywords': keywords, 'summary': summary})
    

    for idx, keys in enumerate(movies['keywords']):
        arr  = []
        for key in keys:
            arr.append(key)
        for a in arr:
            if a not in plot_keywords:
                movies['keywords'][idx].remove(str(a))

    for idx, genres in enumerate(movies['keywords']):
        if genres == []:
            movies.drop(idx,inplace=True)

    movies['summary'] = movies['summary'].apply(lambda x: clean_summary(x))
    movies['summary'] = movies['summary'].apply(lambda x: remove_stopwords(x) if len(x) > BERT_LEN else x)

    movies.reset_index(drop=True, inplace=True)

    for i, summary in enumerate(movies['summary']):
        # books['summary'][i] = summary[:128] + summary[len(summary)-382:] # head + tail
        tail = len(summary) - BERT_LEN
        if tail < 0 :
            tail = 0
        movies['summary'][i] = summary[tail:] # just tail
    
    
    return movies



if __name__ == '__main__':

    path = sys.argv[1]
    for subdir, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(subdir, file), 'r', encoding='utf-8') as f:
                panda = parse(f,file)
                global_movies = pd.concat([global_movies,panda], ignore_index = True)
    
    print(global_movies)
    print(len(global_movies))

    dataset = pd.DataFrame(False, index=np.arange(len(global_movies)),columns=plot_keywords)
    dataset['name'] = global_movies['name'].copy()
    dataset['summary'] = global_movies['summary'].copy()

    dataset.drop(dataset[dataset['name'] == ''].index, inplace=True)
    dataset.drop(dataset[dataset['name'] == None].index, inplace=True)
    dataset.drop(dataset[dataset['name'].isna()].index, inplace=True)
    dataset.drop(dataset[dataset['summary'] == ''].index, inplace=True)
    dataset.drop(dataset[dataset['summary'] == None].index, inplace=True)
    dataset.drop(dataset[dataset['summary'].isna()].index, inplace=True)

    # counts = []
    # for i in range(len(plot_keywords)):
    #     counts.append(0)

    # for genre_idxs in global_movies['keywords']:
    #     tmp = [plot_keywords.index(x) for x in genre_idxs]
    #     for idx in tmp:
    #         counts[idx] = counts[idx] + 1

    # genre_count = {}
    # for idx, genre in enumerate(plot_keywords):
    #     genre_count[str(genre)] = counts[idx]

    # genre_count = {k: v for k, v in sorted(genre_count.items(), key=lambda item: item[1], reverse=True)}

    # for i in genre_count:
    #         print(i,genre_count[str(i)])
    
    # exit()

    for i,genres in enumerate(global_movies['keywords']):
        for genre in genres:
            if str(genre) in dataset.columns:
                dataset.at[i, str(genre)] = True

    dset = Dataset.from_pandas(dataset)

    dset = dset.remove_columns([ '__index_level_0__'])
    dset = dset.filter(lambda example: example['summary'] is not None)


    # 90% train, 10% test + validation
    train_test_valid = dset.train_test_split(test_size=0.1, shuffle=True)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_test_valid['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
        'train': train_test_valid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})


    train_test_valid_dataset.save_to_disk("./processed_imdb_reduced")