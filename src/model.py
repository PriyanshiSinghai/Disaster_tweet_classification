import torch.nn as nn
import torch.nn.functional as F
import torch

class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embed_dim,padding_idx):
        super(BaselineModel,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim,1)
    
    def forward(self,x):
        embedded = self.embedding(x)
        avg_embedded = embedded.mean(dim=1)
        out = self.fc(avg_embedded)
        return torch.sigmoid(out).squeeze(1)

