from transformers import AutoTokenizer, AutoModel, Trainer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
import numpy as np
import os
from datasets import load_from_disk
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, multilabel_confusion_matrix
from transformers import EvalPrediction
from sklearn import metrics

# No interruptions
os.environ["WANDB_DISABLED"] = "true"

reloaded_split_dataset = load_from_disk("./processed_booksumaries_for_roberta_top_genres_only")
# reloaded_split_dataset = load_from_disk("./processed_booksumaries")

labels = [label for label in reloaded_split_dataset['train'].features.keys() if label not in ['summary', 'book_name']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# for i in range(1, len(reloaded_split_dataset['test'])):
#     # print("test ",[id2label[idx] for idx, label in enumerate(reloaded_split_dataset['test'][i]) if label == True])
#     # print("train ",[id2label[idx] for idx, label in enumerate(reloaded_split_dataset['train'][i]) if label == True])
#     arr = []
#     for label in labels:
#         if reloaded_split_dataset['test'][str(label)][-i] == True: 
#             arr.append(label)
#     if arr == []:
#         print(i)
#     print(arr)


model_name = "roberta-base"
max_length = 512

tokenizer = AutoTokenizer.from_pretrained(model_name, padding="max_length", truncation=True, max_length=max_length)

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

device = torch.device('cuda')

model = AutoModelForSequenceClassification.from_pretrained("./best_roberta")
# model = AutoModelForSequenceClassification.from_pretrained("./my_test")


fin_targets=[]
fin_outputs=[]
fin_probs = []
# count = 0
# limit = 10
with torch.no_grad():
    for data in reloaded_split_dataset['train']['summary']:
        encoding = tokenizer(data, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
        outputs = model(**encoding)
        # fin_outputs.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())
        logits = outputs.logits
        # print(logits)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze())
        # print(probs)
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        # print([id2label[idx] for idx, label in enumerate(predictions) if label == 1])
        fin_outputs.append(predictions)
        fin_probs.append(probs)
        # count = count + 1
        # if count == limit:
        #     break
    
    # count = 0
    for data in encoded_dataset['train']['labels']:
        # print(data)
        fin_targets.append(data.tolist())
        # count = count + 1
        # if count == limit:
        #     break
    

    for i,_ in enumerate(fin_targets):
        for j in range(0, len(fin_targets[0])):
            fin_targets[i][j] = int(fin_targets[i][j])
            fin_outputs[i][j] = int(fin_outputs[i][j])


    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro', zero_division=True)
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro', zero_division=True)
    f1_score_samples = metrics.f1_score(fin_targets, fin_outputs, average='samples', zero_division=True)
    f1_score_weighted = metrics.f1_score(fin_targets, fin_outputs, average='weighted', zero_division=True)
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(f"F1 Score (Samples) = {f1_score_samples}")
    print(f"F1 Score (Weighted) = {f1_score_weighted}")
    print("=======================")

    y_pred = np.array(fin_outputs)
    y_true = np.array(fin_targets)

    MR = np.all(y_pred == y_true, axis=1).mean()
    print("MR: ", MR)

    Loss = np.any(y_true != y_pred, axis=1).mean()
    print("Loss: ", Loss)

    def Accuracy(y_true, y_pred):
        temp = 0
        for i in range(y_true.shape[0]):
            temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
        return temp / y_true.shape[0]
    
    print("Accuracy: ", Accuracy(y_true, y_pred))

    def Hamming_Loss(y_true, y_pred):
        temp=0
        for i in range(y_true.shape[0]):
            temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
        return temp/(y_true.shape[0] * y_true.shape[1])
    
    print("Hamming_Loss: ", Hamming_Loss(y_true, y_pred))

    # def F1Measure(y_true, y_pred):
    #     temp = 0
    #     for i in range(y_true.shape[0]):
    #         if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
    #             continue
    #         temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    #     return temp/ y_true.shape[0]
    
    # print("F1(sapmples): ", F1Measure(y_true, y_pred))
from itertools import islice

for i in range(100,110):
    x = {}
    for idx,single_predictions in enumerate(fin_probs[i]):
            x[str(id2label[idx])] = float(single_predictions)

    x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}
    n_items = list(islice(x.items(), 10))
    print("Predicted: ")
    for item in n_items:
        print(item)
    correct_labels = [id2label[idx] for idx, label in enumerate(fin_targets[i]) if label == 1]
    print("Correct: ")
    for label in correct_labels:
        print(label, x[str(label)])
    print("------------------------")
    

    # predicted_labels = [id2label[idx] for idx, label in enumerate(fin_outputs[i]) if label == 1]
    # correct_labels = [id2label[idx] for idx, label in enumerate(fin_targets[i]) if label == 1]
    # print("Predicted: ", predicted_labels)
    # print("Correct: ", correct_labels)

# print(multilabel_confusion_matrix(y_true, y_pred, samplewise=True))
