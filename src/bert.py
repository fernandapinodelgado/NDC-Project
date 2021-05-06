import torch

from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.train()

from transformers import AdamW
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

from main import load_txts
txt_dir = '../TXT_165/'

txts = load_txts(txt_dir)

import spacy
nlp = spacy.load('en_core_web_sm')
raw_text = txts[0]
doc = nlp(raw_text)
sentences = [sent.text.strip() for sent in doc.sents]
print(sentences)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
for sentence in sentences:
    encoding = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    labels = torch.tensor([1, 0, 0]).unsqueeze(0)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

print(outputs.logits)