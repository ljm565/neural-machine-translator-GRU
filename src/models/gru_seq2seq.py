import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import LOGGER, colorstr


class Encoder(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(Encoder, self).__init__()
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        self.hidden_size = config.hidden_dim
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
        self.dropout_layer = nn.Dropout(self.dropout)


    def init_hidden(self):
        h0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size).to(self.device)
        return h0


    def forward(self, x):
        self.batch_size = x.size(0)
        h0 = self.init_hidden()

        x = self.embedding(x)
        x = self.dropout_layer(x)
        x, hn = self.gru(x, h0)
        hn = hn.view(2, -1, self.batch_size, self.hidden_size)
        hn = torch.sum(hn, dim=0)
        return x, hn



class Decoder(nn.Module):
    def __init__(self, config, tokenizer):
        super(Decoder, self).__init__()
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        self.hidden_size = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.use_attention = config.use_attention
        if self.use_attention:
            self.attention = Attention(self.hidden_size)
        self.input_size = self.hidden_size * 2 if self.use_attention else self.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_token_id)
        self.gru = nn.GRU(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)


    def forward(self, x, hidden, enc_output, mask):
        self.batch_size = x.size(0)
        score = None

        x = self.embedding(x)
        if self.use_attention:
            enc_output, score = self.attention(self.relu(enc_output), self.relu(hidden[-1]), mask)
            x = torch.cat((x, enc_output.unsqueeze(1)), dim=-1)
        x = self.dropout_layer(x)
        x, hn = self.gru(x, hidden)
        x = self.fc(self.relu(x))
        return x, hn, score
    

    def batch_inference(self, start_tokens, enc_output, hidden, mask, max_len, tokenizer, loss_func=None, target=None):
        if loss_func:
             assert target != None, LOGGER(colorstr('red', 'Target must be required if you want to return loss values..'))

        decoder_all_output, decoder_all_score = [], []
        trg_word = start_tokens.unsqueeze(1)
        for i in range(max_len):
            dec_output, hidden, score = self.forward(trg_word, hidden, enc_output, mask)
            decoder_all_output.append(dec_output)
            if self.use_attention:
                decoder_all_score.append(score)
            trg_word = torch.argmax(dec_output, dim=-1)

        decoder_all_output = torch.cat(decoder_all_output, dim=1)
        if self.use_attention:
            decoder_all_score = torch.cat(decoder_all_score, dim=2)
        
        predictions = [tokenizer.decode(torch.argmax(pred, dim=-1).tolist()) for pred in decoder_all_output]

        if loss_func:
            loss = loss_func(decoder_all_output[:, :-1, :].reshape(-1, decoder_all_output.size(-1)), target[:, 1:].reshape(-1))
            return predictions, decoder_all_score, loss
        
        return predictions, decoder_all_score, None
        




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
        enc_output = torch.bmm(enc_output, score).squeeze(-1)
        return enc_output, score



