import tensorflow as tf
from transformers import DistilBertTokenizerFast
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
import re
import csv
from tqdm import tqdm

data = []

with open("booksummaries.txt", 'r', encoding="utf8") as f:
    reader = csv.reader(f, dialect='excel-tab', delimiter="\t")
    for row in tqdm(reader):
        data.append(row)

trim = 1000
data = data[:trim]

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

genres = []
for i in books['genre']:
    genres.append(list(json.loads(i).values()))
books['genre'] = genres

all_genres = list(set(sum(genres, [])))
genre_indexes = []
for genres in books['genre']:
    tmp = [all_genres.index(x) for x in genres]
    # tmp += [-1] * (20 - len(tmp))
    arr = []
    for i in range(len(all_genres)):
        arr.append(1 if i in tmp else 0)
    genre_indexes.append(arr)

books['genre_indexes'] = genre_indexes


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

from tensorflow.keras.models import load_model
from transformers import DistilBertTokenizerFast


def multi_label_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """For multi-label classification, one has to define a custom
    acccuracy function because neither tf.keras.metrics.Accuracy nor
    tf.keras.metrics.CategoricalAccuracy evaluate the number of 
    exact matches.

    :Example:
    >>> from tensorflow.keras import metrics
    >>> y_true = tf.convert_to_tensor([[1., 1.]])
    >>> y_pred = tf.convert_to_tensor([[1., 0.]])
    >>> metrics.Accuracy()(y_true, y_pred).numpy()
    0.5
    >>> metrics.CategoricalAccuracy()(y_true, y_pred).numpy()
    1.0
    >>> multi_label_accuracy(y_true, y_pred).numpy()
    0.0
    """   
    y_pred = tf.math.round(y_pred)
    exact_matches = tf.math.reduce_all(y_pred == y_true, axis=1)
    exact_matches = tf.cast(exact_matches, tf.float32)
    return tf.math.reduce_mean(exact_matches)

# Mimicking a production scenario: load the model and tokenizer

BASE_PATH = "./bert_test"
MAX_LENGTH = 512
model = load_model(f"{BASE_PATH}/fine_tuned_test_distilbert", 
                   custom_objects={"multi_label_accuracy": multi_label_accuracy})
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def score_text(text, model=model, tokenizer=tokenizer):
    padded_encodings = tokenizer.encode_plus(
        text,
        max_length=MAX_LENGTH, # truncates if len(s) > max_length
        return_token_type_ids=True,
        return_attention_mask=True,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )
    return model(padded_encodings["input_ids"]).numpy()

score_text("dummy")  # running a dummy prediction as a work-around the extra latency 
# of the first prediction of a loaded TensorFlow model.

text = books['summary'][0]

scores = score_text(text)[0]

scores = pd.Series(scores, all_genres, name="scores")
scores = scores[scores > 0.2]
print(scores.to_frame())
print(books['genre'][0])