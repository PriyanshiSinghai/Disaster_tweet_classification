import pandas as pd
import torch 
from torch.utils.data import DataLoader
from torch import optim, nn

from data_preprocessing import clean_text,tokenize,yield_tokens,TweetDataset,collate_batch
from torchtext.vocab import build_vocab_from_iterator
from model import BaselineModel
from evaluate import evaluate_model

# load data
df = pd.read_csv("/Users/psinghai/Dream_AI/Projects/Disaster_tweets_classification/data/raw/train.csv")
df['clean_text'] = df['text'].apply(clean_text)

# build vocab
vocab = build_vocab_from_iterator(
    yield_tokens(df['clean_text']),
    specials=["<PAD>","<UNK>"]
)
vocab.set_default_index(vocab["<UNK>"])

# Dataset + Dataloader
dataset = TweetDataset(df,vocab)
pad_idx = vocab["<PAD>"]

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: collate_batch(batch,pad_idx)
)

# model
model = BaselineModel(len(vocab),embed_dim=50,padding_idx=pad_idx)

# loss + optimiser
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(),lr=0.001)

# training
def train_model(model,train_loader,criterion,optimiser,epoch=5):
    model.train()
    for epoch in range(epoch):
        total_loss = 0
        for X, y in train_loader:
            optimiser.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred,y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
train_model(model,train_loader,criterion,optimiser,epoch=5)

# evaluate model
evaluate_model(model,train_loader,threshold=0.5)
        
