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

name1 = "-"
data1 = """
Sixty dark years after the victorious but utterly devastating war against an unforeseen alien invasion which left humanity practically on the verge of extinction, most of the remaining earthlings have long since they relocated on Titan, Saturn's largest moon, in 2077. Left behind in the uninhabitable and barren Earth along with a handful of survivors and the extraterrestrial invaders, is the drone technician, Jack Harper, and the communications teammate, Victoria Olsen, who monitor the planet before mankind's final migration to Titan. However, when the plagued with unexplained visions, Jack, rescues the cryptic woman in his dreams, flashbacks of a fragmented memory will soon unearth a startling secret about his mission. In the end, what is the real threat, and above all, who cloaks the truth?
"""

name2 = "-"
data2 = """
When her father unexpectedly dies, young Ella finds herself at the mercy of her cruel stepmother and her scheming stepsisters. Never one to give up hope, Ella's fortunes begin to change after meeting a dashing stranger.

A girl named Ella (Cinderella) has the purest heart living in a cruel world filled with evil stepsisters and an evil stepmother out to ruin Ella's life. Ella becomes one with her pure heart when she meets the Prince and dances her way to a better life with glass shoes, and a little help from her fairy godmother, of course.

A live-action retelling of the classic fairytale about a servant stepdaughter who is abused by her jealous stepmother and stepsisters after her father died. Forced to be a servant in her own house, through it all she did not let anything or anyone crush her spirit. Then one day, she meets a dashing stranger in the woods.
"""
data1 = clean_summary(data1)
if len(data1) > 512:
    data1 = remove_stopwords(data1)
    data1 = data1[len(data1)-512:]

data2 = clean_summary(data2)
if len(data2) > 512:
    data2 = remove_stopwords(data2)
    data2 = data2[len(data2)-512:]

datas = [data1, data2]
predicts = []
probabilities = []
for data in datas:
    with torch.no_grad():
        encoding = tokenizer(data, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
        outputs = model(**encoding)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze())
        floats = []
        predictions = np.zeros(probs.shape)
        x = {}
        for idx,single_predictions in enumerate(probs):
            x[str(id2label[idx])] = float(single_predictions)
            floats.append(float(single_predictions))
        x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}
        n_items = list(islice(x.items(), 15))
        print("Predicted: ")
        for item in n_items:
            print(item)
        
        print("With threshold 0.5: ")
        predictions[np.where(probs >= 0.5)] = 1
        predicts.append(predictions)
        probabilities.append(floats)
        print([id2label[idx] for idx, label in enumerate(predictions) if label == 1])
        print()


from sklearn import metrics
from scipy import stats

print()
print("Correlation:")
print("Cosine similarity from labels: ", metrics.pairwise.cosine_similarity([predicts[0]], [predicts[1]])[0][0])
print("Cosine similarity from probs: ", metrics.pairwise.cosine_similarity([probabilities[0]], [probabilities[1]])[0][0])
print("Separman correlation:", stats.spearmanr(probabilities[0], probabilities[1])[0])
print("Pearson correlation:", stats.pearsonr(probabilities[0], probabilities[1])[0])
