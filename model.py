from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config



class Encoder(nn.Module):
    def __init__(self, config:Config, tokenizer, device):
        super(Encoder, self).__init__()
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_token_id)
        self.gru = nn.GRU(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=True)
        self.dropout = nn.Dropout(self.dropout)


    def init_hidden(self):
        h0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size).to(self.device)
        return h0


    def forward(self, x):
        self.batch_size = x.size(0)
        h0 = self.init_hidden()

        x = self.embedding(x)
        x = self.dropout(x)
        x, hn = self.gru(x, h0)
        hn = hn.view(2, -1, self.batch_size, self.hidden_size)
        hn = torch.sum(hn, dim=0)
        return x, hn



class Decoder(nn.Module):
    def __init__(self, config:Config, tokenizer, device):
        super(Decoder, self).__init__()
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.is_attn = config.is_attn
        self.device = device
        if self.is_attn:
            self.attention = Attention(self.hidden_size)
        self.input_size = self.hidden_size * 2 if self.is_attn else self.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_token_id)
        self.gru = nn.GRU(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout)
        self.dropout = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)


    def forward(self, x, hidden, enc_output, mask):
        self.batch_size = x.size(0)
        score = None

        x = self.embedding(x)
        if self.is_attn:
            enc_output, score = self.attention(self.relu(enc_output), self.relu(hidden[-1]), mask)
            x = torch.cat((x, enc_output.unsqueeze(1)), dim=-1)
        x = self.dropout(x)
        x, hn = self.gru(x, hidden)
        x = self.fc(self.relu(x))
        return x, hn, score



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.enc_dim_changer = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )
        self.enc_wts = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.dec_wts = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.score_wts = nn.Linear(self.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    def forward(self, enc_output, dec_hidden, mask):
        enc_output = self.enc_dim_changer(enc_output)
        score = self.tanh(self.enc_wts(self.relu(enc_output)) + self.dec_wts(dec_hidden).unsqueeze(1))
        score = self.score_wts(score)
        score = score.masked_fill(mask.unsqueeze(2)==0, float('-inf'))
        score = F.softmax(score, dim=1)
        
        enc_output = torch.permute(enc_output, (0, 2, 1))
        enc_output = torch.bmm(enc_output, score).squeeze()
        return enc_output, score



