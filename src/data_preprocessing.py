import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Tokenizer ---
def tokenize(text):
    return text.split()

def yield_tokens(texts):
    for text in texts:
        yield tokenize(text)

# --- Dataset ---
class TweetDataset(Dataset):
    def __init__(self, df, vocab):
        self.texts = df['clean_text'].tolist()
        self.labels = df['target'].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])
        text_tensor = torch.tensor(self.vocab(tokens), dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        return text_tensor, label_tensor

# --- Collate ---
def collate_batch(batch, pad_idx):
    text_list, label_list = [], []
    for text, label in batch:
        text_list.append(text)
        label_list.append(label)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=pad_idx)
    label_list = torch.tensor(label_list, dtype=torch.float)
    return text_list, label_list

