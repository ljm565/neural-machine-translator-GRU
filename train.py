import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import pickle
from tokenizer import Tokenizer
import numpy as np
import time

from config import Config
from utils_func import *
from utils_data import DLoader
from model import Encoder, Decoder



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.data_path = self.config.dataset_path
 
        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.max_len = self.config.max_len

        # define tokenizer
        self.src_tokenizer = Tokenizer(self.config, self.data_path['train'], src=True)
        self.trg_tokenizer = Tokenizer(self.config, self.data_path['train'], src=False)
        self.tokenizers = [self.src_tokenizer, self.trg_tokenizer]

        # dataloader
        torch.manual_seed(999)  # for reproducibility
        if self.mode == 'train':
            self.dataset = {s: DLoader(load_dataset(p), self.tokenizers, self.config) for s, p in self.data_path.items()}
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False)
                for s, d in self.dataset.items()}
        elif self.mode == 'test':
            self.dataset = {s: DLoader(load_dataset(p), self.tokenizers, self.config) for s, p in self.data_path.items() if s == 'test'}
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        self.encoder = Encoder(self.config, self.src_tokenizer, self.device).to(self.device)
        self.decoder = Decoder(self.config, self.trg_tokenizer, self.device).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        if self.mode == 'train':
            self.enc_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
            self.dec_optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.encoder.load_state_dict(self.check_point['model']['encoder'])
                self.decoder.load_state_dict(self.check_point['model']['decoder'])
                self.enc_optimizer.load_state_dict(self.check_point['optimizer']['encoder'])
                self.dec_optimizer.load_state_dict(self.check_point['optimizer']['decoder'])
                del self.check_point
                torch.cuda.empty_cache()
        elif self.mode == 'test' or self.mode == 'inference':
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.encoder.load_state_dict(self.check_point['model']['encoder'])
            self.decoder.load_state_dict(self.check_point['model']['decoder'])
            self.encoder.eval()
            self.decoder.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def train(self):
        early_stop = 0
        best_val_loss = float('inf') if not self.continuous else self.loss_data['best_val_loss']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = [] if not self.continuous else self.loss_data['val_loss_history']
        val_score_history = {'bleu2': [], 'bleu4': [], 'nist2': [], 'nist4': []} if not self.continuous else self.loss_data['val_score_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.encoder.train()
                    self.decoder.train()
                else:
                    self.encoder.eval()
                    self.decoder.eval()

                total_loss = 0
                all_val_trg, all_val_output = [], []
                for i, (src, trg, mask) in enumerate(self.dataloaders[phase]):
                    batch = src.size(0)
                    src, trg = src.to(self.device), trg.to(self.device)
                    if self.config.is_attn:
                        mask = mask.to(self.device)
                    self.enc_optimizer.zero_grad()
                    self.dec_optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        enc_output, hidden = self.encoder(src)
                        
                        teacher_forcing = True if random.random() <= self.config.teacher_forcing_ratio else False
                        decoder_all_output = []
                        for j in range(self.max_len):
                            if teacher_forcing or j == 0 or phase == 'test':
                                trg_word = trg[:, j].unsqueeze(1)
                                dec_output, hidden, _ = self.decoder(trg_word, hidden, enc_output, mask)
                                decoder_all_output.append(dec_output)
                            else:
                                trg_word = torch.argmax(dec_output, dim=-1)
                                dec_output, hidden, _ = self.decoder(trg_word.detach(), hidden, enc_output, mask)
                                decoder_all_output.append(dec_output)

                        decoder_all_output = torch.cat(decoder_all_output, dim=1)
                        loss = self.criterion(decoder_all_output[:, :-1, :].reshape(-1, decoder_all_output.size(-1)), trg[:, 1:].reshape(-1))
                        if phase == 'train':
                            loss.backward()
                            self.enc_optimizer.step()
                            self.dec_optimizer.step()
                        else:
                            all_val_trg.append(trg.detach().cpu())
                            all_val_output.append(decoder_all_output.detach().cpu())

                    total_loss += loss.item()*batch

                    if i % 100 == 0:
                        print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))
                epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
                print('{} loss: {:4f}\n'.format(phase, epoch_loss))

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                if phase == 'test':
                    val_loss_history.append(epoch_loss)

                    # print examples
                    print_samples(src, trg, decoder_all_output, self.tokenizers)

                    # calculate scores
                    all_val_trg, all_val_output = tensor2list(all_val_trg, all_val_output, self.trg_tokenizer)
                    val_score_history['bleu2'].append(cal_scores(all_val_trg, all_val_output, 'bleu', 2))
                    val_score_history['bleu4'].append(cal_scores(all_val_trg, all_val_output, 'bleu', 4))
                    val_score_history['nist2'].append(cal_scores(all_val_trg, all_val_output, 'nist', 2))
                    val_score_history['nist4'].append(cal_scores(all_val_trg, all_val_output, 'nist', 4))
                    print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(val_score_history['bleu2'][-1], val_score_history['bleu4'][-1], val_score_history['nist2'][-1], val_score_history['nist4'][-1]))
                    
                    # save best model
                    early_stop += 1
                    if epoch_loss < best_val_loss:
                        early_stop = 0
                        best_val_loss = epoch_loss
                        best_enc_wts = copy.deepcopy(self.encoder.state_dict())
                        best_dec_wts = copy.deepcopy(self.decoder.state_dict())
                        best_epoch = best_epoch_info + epoch + 1
                        save_checkpoint(self.model_path, [self.encoder, self.decoder], [self.enc_optimizer, self.dec_optimizer])

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val loss: {:4f}, best epoch: {:d}\n'.format(best_val_loss, best_epoch))
        self.model = {'encoder': self.encoder.load_state_dict(best_enc_wts), 'decoder': self.decoder.load_state_dict(best_dec_wts)}
        self.loss_data = {'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history, 'val_score_history': val_score_history}
        return self.model, self.loss_data
    

    def test(self, result_num, model_name):
        if result_num > len(self.dataloaders['test'].dataset):
            print('The number of results that you want to see are larger than total test set')
            raise AssertionError
        
        # statistics of IMDb test set
        phase = 'test'
        total_loss = 0
        all_val_src, all_val_trg, all_val_output, all_val_score = [], [], [], []

        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()

            for src, trg, mask in self.dataloaders[phase]:
                batch = src.size(0)
                src, trg = src.to(self.device), trg.to(self.device)
                if self.config.is_attn:
                    mask = mask.to(self.device)
                enc_output, hidden = self.encoder(src)
                
                decoder_all_output, decoder_all_score = [], []
                for j in range(self.max_len):
                    trg_word = trg[:, j].unsqueeze(1)
                    dec_output, hidden, score = self.decoder(trg_word, hidden, enc_output, mask)
                    decoder_all_output.append(dec_output)
                    if self.config.is_attn:
                        decoder_all_score.append(score)

                decoder_all_output = torch.cat(decoder_all_output, dim=1)
                if self.config.is_attn:
                    decoder_all_score = torch.cat(decoder_all_score, dim=2)
                loss = self.criterion(decoder_all_output[:, :-1, :].reshape(-1, decoder_all_output.size(-1)), trg[:, 1:].reshape(-1))
                all_val_src.append(src.detach().cpu())
                all_val_trg.append(trg.detach().cpu())
                all_val_output.append(decoder_all_output.detach().cpu())
                if self.config.is_attn:
                    all_val_score.append(decoder_all_score.detach().cpu())
                total_loss += loss.item()*batch

        # calculate loss and ppl
        total_loss = total_loss / len(self.dataloaders[phase].dataset)
        print('loss: {}, ppl: {}'.format(total_loss, np.exp(total_loss)))

        # calculate scores
        all_val_trg_l, all_val_output_l = tensor2list(all_val_trg, all_val_output, self.trg_tokenizer)
        bleu2 = cal_scores(all_val_trg_l, all_val_output_l, 'bleu', 2)
        bleu4 = cal_scores(all_val_trg_l, all_val_output_l, 'bleu', 4)
        nist2 = cal_scores(all_val_trg_l, all_val_output_l, 'nist', 2)
        nist4 = cal_scores(all_val_trg_l, all_val_output_l, 'nist', 4)
        print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(bleu2, bleu4, nist2, nist4))

        # visualize the attention score
        all_val_src = torch.cat(all_val_src, dim=0)
        all_val_trg = torch.cat(all_val_trg, dim=0)
        all_val_output = torch.argmax(torch.cat(all_val_output, dim=0), dim=2)
        if self.config.visualize_attn and self.config.is_attn:
            all_val_score = torch.cat(all_val_score, dim=0)
            save_path = self.base_path + 'result/' + model_name
            visualize_attn(all_val_score, all_val_src, all_val_trg, all_val_output, self.tokenizers, result_num, save_path)
        else:
            ids = random.sample(list(range(all_val_trg.size(0))), result_num)
            print_samples(all_val_src, all_val_trg, all_val_output, self.tokenizers, result_num, ids)

    
    def inference(self, query):
        query, mask = make_inference_data(query, self.tokenizers[0], self.max_len)

        with torch.no_grad():
            query = query.to(self.device)
            if self.config.is_attn:
                mask = mask.to(self.device)
            self.encoder.eval()
            self.decoder.eval()

            enc_output, hidden = self.encoder(query)
            decoder_all_output, decoder_sos = [], torch.LongTensor([[self.tokenizers[1].sos_token_id]]).to(self.device)
            for j in range(self.max_len):
                if j == 0:
                    dec_output, hidden, _ = self.decoder(decoder_sos, hidden, enc_output, mask)
                    decoder_all_output.append(dec_output)
                else:
                    trg_word = torch.argmax(dec_output, dim=-1)
                    dec_output, hidden, _ = self.decoder(trg_word.detach(), hidden, enc_output, mask)
                    decoder_all_output.append(dec_output)
            decoder_all_output = torch.cat(decoder_all_output, dim=1)
            output = self.tokenizers[1].decode(torch.argmax(decoder_all_output.detach().cpu(), dim=-1)[0].tolist())
        
        if output.split()[-1] == self.tokenizers[1].eos_token:
            return ' '.join(output.split()[:-1])
        return output       