import re
import os
import random
import unicodedata
import matplotlib.pyplot as plt

import torch

from utils import LOGGER, colorstr



def preprocessing(s):
    s = unicodeToAscii(s)
    for punc in '!?.,"':
        s = s.replace(punc, ' '+punc)
    s = re.sub('[#$%&()*+\-/:;<=>@\[\]^_`{|}~]', '', s).lower()
    s = ' '.join(s.split())
    return s


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def print_samples(source, target, prediction):
    LOGGER.info('\n' + '-'*100)
    LOGGER.info(colorstr('SRC       : ') + source)
    LOGGER.info(colorstr('GT        : ') + target)
    LOGGER.info(colorstr('Prediction: ') + prediction)
    LOGGER.info('-'*100 + '\n')


def visualize_attn(data4vis, tokenizers, result_num, save_dir):
    src_tokenizer, trg_tokenizer = tokenizers[0], tokenizers[1]
    src, trg, pred, score = data4vis['src'], data4vis['trg'], data4vis['pred'], torch.cat(data4vis['score'], dim=0)
    ids = random.sample(range(len(src)), result_num)

    for num, i in enumerate(ids):
        src_tok = src_tokenizer.tokenize(src[i])
        trg_tok = trg_tokenizer.tokenize(trg[i])
        pred_tok = trg_tokenizer.tokenize(pred[i])
        
        src_st, src_tr = 1, len(src_tok) - 1
        trg_st, trg_tr = 1, len(trg_tok) - 1
        pred_st, pred_tr = 0, len(pred_tok) - 1

        score_i = score[i, src_st:src_tr, pred_st:pred_tr]
        src_tok = src_tok[src_st:src_tr]
        trg_tok = trg_tok[trg_st:trg_tr]
        pred_tok = pred_tok[pred_st:pred_tr]

        plt.figure(figsize=(8, 8))
        plt.title('Neural Machine Translator Attention', fontsize=20)
        plt.imshow(score_i, cmap='gray')
        plt.yticks(list(range(len(src_tok))), src_tok)
        plt.xticks(list(range(len(pred_tok))), pred_tok, rotation=90)
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f'attention_{num}.png'))

        print_samples(' '.join(src_tok), ' '.join(trg_tok), ' '.join(pred_tok))