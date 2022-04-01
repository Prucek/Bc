import torch

import pandas as pd
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import csv
from tqdm import tqdm
from sklearn import metrics

from transformers import BertTokenizer, BertModel

data = []

with open("booksummaries.txt", 'r', encoding="utf8") as f:
    reader = csv.reader(f, dialect='excel-tab', delimiter="\t")
    for row in tqdm(reader):
        data.append(row)

# trim = 1000
# data = data[:trim]

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
all_genres.sort()

genre_indexes = []
for genres in books['genre']:
    tmp = [all_genres.index(x) for x in genres]
    arr = []
    for i in range(len(all_genres)):
        arr.append(1 if i in tmp else 0)
    genre_indexes.append(arr)

books['genre_indexes'] = genre_indexes

# print(len(all_genres)) 227
# exit()


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

dataset = pd.DataFrame(0, index=np.arange(len(data)),columns=all_genres)
dataset['book_name'] = books['book_name'].copy()
dataset['summary'] = books['summary'].copy()

for i,genres in enumerate(books['genre']):
    for genre in genres:
        if str(genre) in dataset.columns:
            dataset[str(genre)][i] = 1


ckpt_path = "./curr_ckpt"
best_model_path = "./best_model.pt"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-05
device = torch.device('cuda')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['summary']
        self.targets = self.df[all_genres].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }

train_size = 0.8
train_df = dataset.sample(frac=train_size,random_state=200)
val_df = dataset.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)


train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
valid_dataset = CustomDataset(val_df, tokenizer, MAX_LEN)

train_data_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)



class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        # self.bert_model  = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", problem_type="multi_label_classification")
        # self.bert_model = ReformerForSequenceClassification.from_pretrained("google/reformer-crime-and-punishment", problem_type="multi_label_classification")
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, len(all_genres))
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

model = BERTClass()
model.to(device)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)



def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min

model, optimizer, checkpoint, loss = load_ckp(best_model_path, model, optimizer)
model.eval()

# print(model)
# print(optimizer)
# print(checkpoint)
# print(loss* len(val_data_loader))


# example = dataset['summary'][0]

# print(dataset['book_name'][0]," expected: ",books['genre'][0])

fin_targets=[]
fin_outputs=[]
with torch.no_grad():
    for _, data in enumerate(val_data_loader, 0):
        input_ids = data['input_ids'].to(device, dtype=torch.long)
        attention_mask = data['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        
        output = model(input_ids, attention_mask, token_type_ids)
        fin_outputs.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())

        targets = data['targets'].to(device, dtype = torch.float)
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
    
 
    print("====================Validation===========================")

    for i,_ in enumerate(fin_targets):
        for j in range(0, len(fin_targets[0])):
            fin_targets[i][j] = int(fin_targets[i][j])
            fin_outputs[i][j] = int(fin_outputs[i][j] >= 0.1)

    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro')
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro')
    f1_score_samples = metrics.f1_score(fin_targets, fin_outputs, average='samples')
    f1_score_weighted = metrics.f1_score(fin_targets, fin_outputs, average='weighted')
    f1_score_mlc = metrics.f1_score(fin_targets, fin_outputs, zero_division=1, average=None)
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(f"F1 Score (Samples) = {f1_score_samples}")
    print(f"F1 Score (zerodiv) = {f1_score_weighted}")
    print(f"F1 Score (mlc) = {f1_score_mlc}")
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

    def Recall(y_true, y_pred):
        temp = 0
        for i in range(y_true.shape[0]):
            if sum(y_pred[i]) == 0:
                continue
            temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
        return temp/ y_true.shape[0]
    
    print("Recall: ", Recall(y_true, y_pred))

    def F1Measure(y_true, y_pred):
        temp = 0
        for i in range(y_true.shape[0]):
            if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
                continue
            temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
        return temp/ y_true.shape[0]
    
    print("F1: ", F1Measure(y_true, y_pred))
# print("predicted:")
# for i, out in enumerate(final_output[0]):
#     if out > 0.1:
#         print(all_genres[i])

# testing
# example = dataset['summary'][0]

# print(dataset['book_name'][0]," expected: ",books['genre'][0])
# encodings = tokenizer.encode_plus(
#     example,
#     None,
#     add_special_tokens=True,
#     max_length=MAX_LEN,
#     padding='max_length',
#     return_token_type_ids=True,
#     truncation=True,
#     return_attention_mask=True,
#     return_tensors='pt'
# )
# with torch.no_grad():
#     input_ids = encodings['input_ids'].to(device, dtype=torch.long)
#     attention_mask = encodings['attention_mask'].to(device, dtype=torch.long)
#     token_type_ids = encodings['token_type_ids'].to(device, dtype=torch.long)
#     output = model(input_ids, attention_mask, token_type_ids)
#     final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
#     print("predicted:")
#     for i, out in enumerate(final_output[0]):
#         if out > 0.1:
#             print(all_genres[i])

# fin_targets=[]
# fin_outputs=[]
# with torch.no_grad():
#     for _, data in enumerate(val_data_loader, 0):
#         input_ids = data['input_ids'].to(device, dtype=torch.long)
#         attention_mask = data['attention_mask'].to(device, dtype=torch.long)
#         token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        
#         output = model(input_ids, attention_mask, token_type_ids)
#         fin_outputs.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())

#         targets = data['targets'].to(device, dtype = torch.float)
#         fin_targets.extend(targets.cpu().detach().numpy().tolist())