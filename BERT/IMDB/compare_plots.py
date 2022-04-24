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
After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey to bring him home.

A clown fish named Marlin lives in the Great Barrier Reef and loses his son, Nemo, after he ventures into the open sea, despite his father's constant warnings about many of the ocean's dangers. Nemo is abducted by a boat and netted up and sent to a dentist's office in Sydney. While Marlin ventures off to try to retrieve Nemo, Marlin meets a fish named Dory, a blue tang suffering from short-term memory loss. The companions travel a great distance, encountering various dangerous sea creatures such as sharks, anglerfish and jellyfish, in order to rescue Nemo from the dentist's office, which is situated by Sydney Harbour. While the two are searching the ocean far and wide, Nemo and the other sea animals in the dentist's fish tank plot a way to return to the sea to live their lives free again.
"""

name2 = "-"
data2 = """
A retired CIA agent travels across Europe and relies on his old skills to save his estranged daughter, who has been kidnapped while on a trip to Paris.

Seventeen year-old Kim is the pride and joy of her father Bryan Mills. Bryan is a retired agent who left the Central Intelligence Agency to be near Kim in California. Kim lives with her mother Lenore and her wealthy stepfather Stuart. Kim manages to convince her reluctant father to allow her to travel to Paris with her friend Amanda. When the girls arrive in Paris they share a cab with a stranger named Peter, and Amanda lets it slip that they are alone in Paris. Using this information an Albanian gang of human traffickers kidnaps the girls. Kim barely has time to call her father and give him information. Her father gets to speak briefly to one of the kidnappers and he promises to kill the kidnappers if they do not let his daughter go free. The kidnapper wishes him "good luck," so Bryan Mills travels to Paris to search for his daughter and her friend.
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
print("Cosine similarity from probs: ", metrics.pairwise.cosine_similarity([probabilities[0]], [probabilities[1]])[0][0])
print("Spearman correlation:", stats.spearmanr(probabilities[0], probabilities[1])[0])
print("Pearson correlation:", stats.pearsonr(probabilities[0], probabilities[1])[0])
