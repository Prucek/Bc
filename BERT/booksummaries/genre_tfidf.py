import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
import re
import csv
from tqdm import tqdm

"""
    Read book summaries
"""
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

"""
    Add to panda data frame
"""
books = pd.DataFrame({'book_id': book_id, 'book_name': book_name,
                       'genre': genre, 'summary': summary})


books.drop(books[books['genre'] == ''].index, inplace=True)
books.drop(books[books['genre'] == None].index, inplace=True)
books.drop(books[books['summary'] == ''].index, inplace=True)
books.drop(books[books['summary'] == None].index, inplace=True)
books.drop(books[books['genre'].isin([])].index, inplace=True)
books.drop(books[books['summary'].isin([])].index, inplace=True)

"""
    Extract generes
"""
genres = []
for i in books['genre']:
    genres.append(list(json.loads(i).values()))
books['genre'] = genres

all_genres = list(set(sum(genres, [])))


"""
    Make all lowercase
"""
def clean_summary(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


"""
    Remove unimportant words
"""
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


books['summary'] = books['summary'].apply(lambda x: clean_summary(x))
books['summary'] = books['summary'].apply(lambda x: remove_stopwords(x))

books.reset_index(drop=True, inplace=True)


multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(books['genre'])

# transform target variable
y = multilabel_binarizer.transform(books['genre'])

# split dataset into training and validation set
xtrain, xval, ytrain, yval  = train_test_split(books['summary'], y, test_size=0.2)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8)
# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score, accuracy_score
lr = LogisticRegression()
clf = OneVsRestClassifier(lr)
# fit model on train data
clf.fit(xtrain_tfidf, ytrain)
# make predictions for validation set
y_pred = clf.predict(xval_tfidf)

def Accuracy(y_true, y_pred):
        temp = 0
        for i in range(y_true.shape[0]):
            temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
        return temp / y_true.shape[0]

# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)
t = 0.25  # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)
# evaluate performance

print("Accuracy: ", Accuracy(yval, y_pred_new))
accuracy = accuracy_score(yval, y_pred_new)
f1_score_micro = f1_score(yval, y_pred_new, average='micro', zero_division=True)
f1_score_macro = f1_score(yval, y_pred_new, average='macro', zero_division=True)
f1_score_samples = f1_score(yval, y_pred_new, average='samples', zero_division=True)
f1_score_weighted = f1_score(yval, y_pred_new, average='weighted', zero_division=True)
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")
print(f"F1 Score (Samples) = {f1_score_samples}")
print(f"F1 Score (Weighted) = {f1_score_weighted}")
print("=======================")


def infer_tags(q):
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)


for i in range(50):
    k = xval.sample(1).index[0]
    print("Book: ", books['book_name'][k], "\nPredicted genre: ", infer_tags(xval[k]))
    print("Actual genre: ", books['genre'][k], "\n")

