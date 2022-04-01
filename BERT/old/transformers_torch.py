import torch
import pandas as pd
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import csv
from tqdm import tqdm
import shutil
import sys
from sklearn import metrics

from transformers import BertTokenizer, BertModel, ReformerTokenizer, ReformerForSequenceClassification
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
books.drop(books[books['genre'] == None].index, inplace=True)
books.drop(books[books['summary'] == ''].index, inplace=True)
books.drop(books[books['summary'] == None].index, inplace=True)


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

# print(all_genres)
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

for i, summary in enumerate(books['summary']):
    books['summary'][i] = summary[:128] + summary[len(summary)-382:]

# print(len(books['summary'][0]))
# exit()

# hyperparameters
MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-05



# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# tokenizer = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")


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
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        # self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True, problem_type="multi_label_classification")
        self.bert_model  = BertModel.from_pretrained("roberta-base", return_dict=True, problem_type="multi_label_classification")
        # self.bert_model  = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", problem_type="multi_label_classification")
        # self.bert_model = ReformerForSequenceClassification.from_pretrained("google/reformer-crime-and-punishment", problem_type="multi_label_classification")
        self.dropout = torch.nn.Dropout(0.2)
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

device = torch.device('cuda')
model = BERTClass()
model.to(device)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def validate(fin_targets, fin_outputs):
    for i,_ in enumerate(fin_targets):
        for j in range(0, len(fin_targets[0])):
            fin_targets[i][j] = int(fin_targets[i][j])
            fin_outputs[i][j] = int(fin_outputs[i][j] >= 0.1)

    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_weighted = metrics.f1_score(fin_targets, fin_outputs, average='weighted', zero_division=True)
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (weighted) = {f1_score_weighted}")

    print("=======================")

    y_pred = np.array(fin_outputs)
    y_true = np.array(fin_targets)

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




def train_model(n_epochs, training_loader, validation_loader, model, 
                optimizer, checkpoint_path, best_model_path):
   
    valid_loss_min = np.Inf
   
    for epoch in range(1, n_epochs+1):
        train_loss = 0
        valid_loss = 0
        val_targets=[]
        val_outputs=[]

        model.train()
        print('############# Epoch {}: Training Start   #############'.format(epoch))
        for batch_idx, data in enumerate(training_loader):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()

            loss = loss_fn(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
    
        print('############# Epoch {}: Training End     #############'.format(epoch))
        
        print('############# Epoch {}: Validation Start   #############'.format(epoch))
    
        model.eval()
   
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['input_ids'].to(device, dtype = torch.long)
                mask = data['attention_mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            validate(val_targets,val_outputs)

        print('############# Epoch {}: Validation End     #############'.format(epoch))

        print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss
                ))
        
        # create checkpoint variable and add important data
        checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
        }
            
            # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
            
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

        print('############# Epoch {}  Done   #############\n'.format(epoch))

    return model



ckpt_path = "./curr_ckpt"
best_model_path = "./best_model.pt"

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, ckpt_path, best_model_path)
