import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import  DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

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

data = data[:50]

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

"""
    Extract generes
"""
genres = []
for i in books['genre']:
    genres.append(list(json.loads(i).values()))
books['genre'] = genres

all_genres = list(set(sum(genres, [])))
genre_indexes = []
for genres in books['genre']:
    tmp = [all_genres.index(x) for x in genres]
    arr = []
    for i in range(len(all_genres)):
        arr.append(1 if i in tmp else 0)
    genre_indexes.append(arr)

books['genre_indexes'] = genre_indexes


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

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


books['summary'] = books['summary'].apply(lambda x: clean_summary(x))
books['summary'] = books['summary'].apply(lambda x: remove_stopwords(x))

train_texts = list(books['summary'][:int(len(books['summary'])*0.8)])
train_labels = list(books['genre_indexes'][:int(len(books['genre_indexes'])*0.8)])
test_texts = list(books['summary'][int(len(books['summary'])*0.8)+1:])
test_labels = list(books['genre_indexes'][int(len(books['genre_indexes'])*0.8)+1:])

# split dataset into training and validation set
train_texts, valid_texts, train_labels, valid_labels = \
train_test_split(train_texts, train_labels, test_size=0.2, random_state = 0)

"""
    BERT
"""
model_name = "bert-base-uncased"
#model_name = 'distilbert-base-uncased'
# max sequence length for each document/sentence sample
max_length = 512
# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)


train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
valid_encodings = tokenizer(valid_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
test_encodings  = tokenizer(test_texts,  padding=True, truncation=True, return_tensors="pt", max_length=max_length)



class BooksSummaryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = BooksSummaryDataset(train_encodings, train_labels)
valid_dataset = BooksSummaryDataset(valid_encodings, valid_labels)
test_dataset  = BooksSummaryDataset(test_encodings,  test_labels)



tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(all_genres))

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(valid_dataset, batch_size=8)

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()