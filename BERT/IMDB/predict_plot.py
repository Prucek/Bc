from transformers import AutoTokenizer, AutoModel, Trainer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
from datasets import load_from_disk
import re
import nltk
from nltk.corpus import stopwords
from itertools import islice

# No interruptions
os.environ["WANDB_DISABLED"] = "true"
nltk.download('stopwords')

reloaded_split_dataset = load_from_disk("./processed_imdb_reduced")
# reloaded_split_dataset = load_from_disk("./processed_booksumaries")

labels = [label for label in reloaded_split_dataset['valid'].features.keys() if label not in ['summary', 'name']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


model_name = "roberta-base"
max_length = 512

tokenizer = AutoTokenizer.from_pretrained(model_name, padding="max_length", truncation=True, max_length=max_length)


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

model = AutoModelForSequenceClassification.from_pretrained("./best_roberta_imdb_30_epochs_reduced")

name = "Harry Potter and the Sorcerer's Stone"
data = """
Among the seminal texts of the 20th century, Nineteen Eighty-Four is a rare work that grows more haunting as its futuristic purgatory becomes more real. Published in 1949, the book offers political satirist George Orwell's nightmarish vision of a totalitarian, bureaucratic world and one poor stiff's attempt to find individuality. The brilliance of the novel is Orwell's prescience of modern life—the ubiquity of television, the distortion of the language—and his ability to construct such a thorough version of hell. Required reading for students since it was published, it ranks among the most terrifying novels ever written. 
"""
data = clean_summary(data)
# if len(data) > 512:
data = remove_stopwords(data)
data = data[len(data)-512:]

print(name)

with torch.no_grad():
    encoding = tokenizer(data, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
    outputs = model(**encoding)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze())
    predictions = np.zeros(probs.shape)
    x = {}
    for idx,single_predictions in enumerate(probs):
        x[str(id2label[idx])] = float(single_predictions)
    x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}
    n_items = list(islice(x.items(), 15))
    print("Predicted: ")
    for item in n_items:
        print(item)
    
    print("With threshold 0.5: ")
    predictions[np.where(probs >= 0.5)] = 1
    print([id2label[idx] for idx, label in enumerate(predictions) if label == 1])