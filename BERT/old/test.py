import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("train.csv")
texts = list(dataset["comment_text"])
label_names = dataset.drop(["id", "comment_text"], axis=1).columns
labels = dataset[label_names].values

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

print(len(train_labels))
print(train_labels)
print(train_labels[20])
print(type(train_labels))
print(train_labels.shape[1])