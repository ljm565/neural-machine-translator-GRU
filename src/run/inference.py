import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from trainer import Trainer
from utils import LOGGER, colorstr
from utils.training_utils import choose_proper_resume_model



def env_setup():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):    
    # init config
    config = load_config(os.path.join(args.resume_model_dir, 'args.yaml'))
    
    # init environment
    env_setup()

    # validation 
    inference(args, config)

    
def inference(args, config):
    if config.device == 'mps':
        LOGGER.warning(colorstr('yellow', 'cpu is automatically selected because mps leads inaccurate validation.'))
        device = torch.device('cpu')
    else:
        device = torch.device('cpu') if config.device == 'cpu' else torch.device(f'cuda:{config.device[0]}')
        
    trainer = Trainer(
        config, 
        'validation', 
        device, 
        resume_path=choose_proper_resume_model(args.resume_model_dir, args.load_model_type) if args.resume_model_dir else None
    )

    LOGGER.info(colorstr('inference starting...\n'))
    while 1:
        query = input('English: ')
        if query == 'exit':
            break
        output = trainer.inference(query)
        print('French: ', output, '\n')




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--resume_model_dir', type=str, required=False)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    args = parser.parse_args()
    
    main(args)

    