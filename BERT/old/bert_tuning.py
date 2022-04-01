import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizerFast, DistilBertConfig
from transformers import TFDistilBertForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset

import pandas as pd
import json
import numpy as np  
import nltk
import math
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


# dataset = Dataset.from_pandas(books)
# split_dataset = dataset.train_test_split(test_size=0.1)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    list(books['summary']), np.array(list(books['genre_indexes'].values)), test_size=0.4, random_state=42
)

#print(f"Labels: {pd.Series(train_labels[23], all_genres).to_dict()}")

# print(len(train_texts))
# print(train_texts[0])
# print(len(train_labels))
# print(train_labels[0])
# print(type(train_labels))
# print(train_labels.shape[1])
# exit()

"""
    Plot for word count
"""
# import seaborn as sns

# text_lengths = [len(t.split()) for t in train_texts]
# ax = sns.histplot(data=text_lengths, kde=True, stat="density")
# ax.set_title("Texts length distribution (number of words)")

# import matplotlib.pyplot as plt
# plt.show()
# exit()

import numpy as np  
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, average_precision_score

pd.set_option("display.precision", 3)

# dummy = DummyClassifier(strategy="prior")
# dummy.fit(train_texts, train_labels)
# y_pred = dummy.predict(test_texts)
# y_prob = dummy.predict_proba(test_texts)
# y_prob = np.array(y_prob)[:, :, 1].T

def compute_metrics(y_true: np.array, y_prob: np.array) -> pd.Series:
    """Compute several performance metrics for multi-label classification. """
    y_pred = y_prob.round()
    metrics = dict()
    metrics["Multi-label accuracy"] = np.all(y_pred == y_true, axis=1).mean()
    metrics["Binary accuracy"] = (y_pred == y_true).mean()
    metrics["Loss"] = log_loss(y_true, y_prob)
    metrics["Average Precision"] = average_precision_score(y_true, y_prob)
    return pd.Series(metrics)

# evaluation = compute_metrics(test_labels, y_prob).to_frame(name="Dummy")

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
# vectorizer.fit(train_texts)
# vectorizer.fit(test_texts)
# x_train = vectorizer.transform(train_texts)
# y_train = train_labels
# x_test = vectorizer.transform(test_texts)
# y_test = test_labels


# # using binary relevance
# from skmultilearn.problem_transform import BinaryRelevance
# from sklearn.naive_bayes import GaussianNB
# # initialize binary relevance multi-label classifier
# # with a gaussian naive bayes base classifier
# classifier = BinaryRelevance(GaussianNB())
# # train
# classifier.fit(x_train, y_train)
# # predict
# predictions = classifier.predict(x_test)
# print("Accuracy = ",accuracy_score(y_test,predictions))

# exit()

"""
    BERT
"""

model_name = 'distilbert-base-uncased'
max_length = 512

config = DistilBertConfig.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, problem_type="multi_label_classification")
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# def tokenize_dataset(dataset):
#     return tokenizer(
#         dataset["summary"],
#         max_length=max_length,
#         truncation=True,
#     )

# tokenized_dataset = split_dataset.map(tokenize_dataset, batched=True)

# train_dataset = tokenized_dataset["train"].to_tf_dataset(
#     columns=["input_ids", "attention_mask", "token_type_ids"],
#     label_cols=["genre_indexes"],
#     shuffle=True,
#     collate_fn=data_collator,
#     batch_size=8)

# validation_dataset = tokenized_dataset["test"].to_tf_dataset(
#     columns=["input_ids", "attention_mask", "token_type_ids"],
#     label_cols=["genre_indexes"],
#     shuffle=True,
#     collate_fn=data_collator,
#     batch_size=8)


train_encodings = tokenizer(train_texts, truncation=True, padding=True, 
                            max_length=max_length, return_tensors="tf")
test_encodings = tokenizer(test_texts, truncation=True, padding=True, 
                           max_length=max_length, return_tensors="tf")

# Create TensorFlow datasets to feed the model for training and evaluation
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))

# model = TFAutoModelForSequenceClassification.from_pretrained(model_name, \
#     problem_type="multi_label_classification", num_labels=len(all_genres))

from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
transformer_model = TFDistilBertForSequenceClassification.from_pretrained(
    model_name, output_hidden_states=False
)

bert = transformer_model.layers[0]

# The input is a dictionary of word identifiers 
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
inputs = {'input_ids': input_ids}

# Here we select the representation of the first token ([CLS]) for classification
# (a.k.a. "pooled representation")
bert_model = bert(inputs)[0][:, 0, :] 

# Add a dropout layer and the output layer
dropout = Dropout(config.dropout, name='pooled_output')
pooled_output = dropout(bert_model, training=False)
output = Dense(
    units=train_labels.shape[1],
    kernel_initializer=TruncatedNormal(stddev=config.initializer_range), 
    activation="sigmoid",  # Choose a sigmoid for multi-label classification
    name='output'
)(pooled_output)

model = Model(inputs=inputs, outputs=output, name='BERT_MultiLabel')

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=tf.metrics.SparseCategoricalAccuracy(),
# )

# model.fit(train_dataset, validation_data=validation_dataset, epochs=3)

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

loss = BinaryCrossentropy()
optimizer = Adam(5e-5)
metrics = [
    AUC(name="average_precision", curve="PR", multi_label=True)
]
 
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
training_history = model.fit(
    train_dataset.shuffle(1000).batch(16), epochs=2, batch_size=16, 
    validation_data=test_dataset.batch(16)
)

benchmarks = model.evaluate(
    test_dataset.batch(16), return_dict=True, batch_size=16
)
evaluation = [
    benchmarks[k] for k in 
    [ "loss", "average_precision"]
]

print(evaluation)

import os

BASE_PATH = "./bert_test"

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

model.save(f"{BASE_PATH}/fine_tuned_test_distilbert")
