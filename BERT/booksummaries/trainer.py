from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_from_disk
import torch
import numpy as np
import os

# No interruptions
os.environ["WANDB_DISABLED"] = "true"

reloaded_split_dataset = load_from_disk("./processed_booksumaries_50+")

labels = [label for label in reloaded_split_dataset['train'].features.keys() if label not in ['summary', 'book_name']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


# print(split_dataset)
# example = split_dataset['train'][0]
# print(example)
# print(split_dataset['train'].column_names)
# exit()

model_name = 'roberta-base'
max_length = 512

tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_length, truncation=True)

def preprocess_data(examples):
    # take a batch of texts
    text = examples["summary"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()
    
    return encoding

encoded_dataset = reloaded_split_dataset.map(preprocess_data, batched=True, remove_columns=reloaded_split_dataset['train'].column_names)

encoded_dataset.set_format("torch")

# example = encoded_dataset['test'][0]
# print(example['labels'])
# print([id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0])
# exit()

model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)



batch_size = 8
metric_name = "f1"

from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction

args = TrainingArguments(
    f"roberta-finetuned-50+",
    evaluation_strategy = "steps",
    eval_steps = 500,
    save_strategy = "steps",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    warmup_steps= 10000,
    adam_beta1= 0.9,
    adam_beta2= 0.999,
    adam_epsilon= 1e-08,
    gradient_accumulation_steps=4
)

def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.3):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_average = f1_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=True)
    accuracy_all = accuracy_score(y_true, y_pred)
    accuracy = Accuracy(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_average,
               'accuracy_all': accuracy_all,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()



count = 0
for book in reloaded_split_dataset['valid']:
    print("name: ", book['book_name'])
    arr = []
    for label in labels:
        if book[str(label)] == 1:
            arr.append(label)
    print("correct: ", arr)

    encoding = tokenizer(book['summary'], return_tensors="pt", max_length=max_length)
    encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

    outputs = trainer.model(**encoding)
    logits = outputs.logits

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.3)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    print("predicted: ", predicted_labels)
    count = count + 1
    if count > 10:
        break

trainer.save_model("./best_roberta_50+")