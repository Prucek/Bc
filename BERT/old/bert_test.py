#!/usr/bin/env python
# trying out bert
# Author: Peter Rucek
# Date: 07.11.2021
# Usage: python3 bert_test.py <path_to_directory_where_downloaded_html_are_located>

import torch
from transformers import XLNetTokenizer, XLNetModel, AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# model = XLNetModel.from_pretrained('xlnet-base-cased')

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
# outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

# last_hidden_states = outputs.last_hidden_state
# token_logits = model(input, return_dict=True).logits

# print(last_hidden_states)
# print(int(torch.argmax(outputs.logits))+1)

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokens = tokenizer.encode("Jordan Peterson may be the only clinical psychologist who believes that psychology is subordinate to philosophy and the one thing that psychology and philosophy both genuflect before is story. Story, or myth, predates religion and is, in fact, as old as language itself.   In his earlier book, Maps of Meaning: The Architecture of Belief, Peterson connects the stories we share with our earliest ancestors with modern knowledge of behavior and the mind. It’s a textbook for his popular University of Toronto courses.  The one-time dish washer and mill worker spent nearly 20 years at the University before garnering international attention. In September 2016, Peterson released a couple of videos opposing an amendment to the Canadian Human Rights Act which he contended could send someone to jail for refusing to use a made-up gender identity pronoun. Peterson went on to testify before the Canadian Senate, and has emerged as a foremost critic of postmodernism on North American campuses.  Postmodernism is the “new skin of communism,” In Peterson’s view. The ideology has been so thoroughly discredited from an economic standpoint that those who still advocate for it, for either political or emotional reasons, have resorted to attacking the very process in which something can be discredited—reason and debate. At the same time they have worked to change the face of oppression away from those living in poverty toward individuals who don’t look or act like those who hold most of the positions of power and authority in Western society.   Peterson’s classroom is now the entire globe. Millions are watching his lectures and other videos on YouTube. For this new and greater audience, a more accessible, more affordable compendium than Maps of Meaning was called for.   12 Rules for Life: An Antidote to Chaos is more affordable for sure, but only slightly more accessible. Part self-help book, part memoir, part Maps for the masses, it’s organized sprawlingly.   (Read full review at https://simplicityandpurity.wordpress...) ", return_tensors='pt')
result = model(tokens)
print(int(torch.argmax(result.logits))+1)