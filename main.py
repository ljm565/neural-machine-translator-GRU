import torch
import pickle
from argparse import ArgumentParser
import os
from train import Trainer
from config import Config
import json
import sys

from preprocessing import preprocessing_data



def main(config_path:Config, args:ArgumentParser):
    device = torch.device('cuda:0') if args.device == 'gpu' else torch.device('cpu')
    print('Using {}'.format(device))

    if (args.cont and args.mode == 'train') or args.mode != 'train':
        try:
            config = Config(config_path)
            config = Config(config.base_path + '/model/' + args.name + '/' + args.name + '.json')
            base_path = config.base_path
        except:
            print('*'*36)
            print('There is no [-n, --name] argument')
            print('*'*36)
            sys.exit()
    else:
        config = Config(config_path)
        base_path = config.base_path
        dataset_path = {}

        # make neccessary folders
        os.makedirs(base_path+'model', exist_ok=True)
        os.makedirs(base_path+'loss', exist_ok=True)
        os.makedirs(base_path+'data/processed', exist_ok=True)

        # download IMDb review data
        if not (os.path.isfile(base_path+'data/processed/eng-fra.train') and os.path.isfile(base_path+'data/processed/eng-fra.test')):
            print('Processing the raw eng-fra data')
            trainset, testset = preprocessing_data(base_path+'data/raw/eng-fra.txt')
            for split, dataset in zip(['train', 'test'], [trainset, testset]):
                with open(base_path+'data/processed/eng-fra.'+split, 'wb') as f:
                    pickle.dump(dataset, f)

        for split in ['train', 'test']:
            dataset_path[split] = base_path+'data/processed/eng-fra.'+split
        config.dataset_path = dataset_path

        # redefine config
        config.loss_data_path = base_path + 'loss/' + config.loss_data_name + '.pkl'

        # make model related files and folder
        model_folder = base_path + 'model/' + config.model_name
        config.model_path = model_folder + '/' + config.model_name + '.pt'
        model_json_path = model_folder + '/' + config.model_name + '.json'
        os.makedirs(model_folder, exist_ok=True)
          
        with open(model_json_path, 'w') as f:
            json.dump(config.__dict__, f)
    
   
    trainer = Trainer(config, device, args.mode, args.cont)

    if args.mode == 'train':
        loss_data_path = config.loss_data_path
        print('Start training...\n')
        models, loss_data = trainer.train()

        print('Saving the loss related data...')
        with open(loss_data_path, 'wb') as f:
            pickle.dump(loss_data, f)

    elif args.mode == 'test':
        print('test starting...\n')
        os.makedirs(base_path+'result', exist_ok=True)
        trainer.test(config.result_num, args.name)
    
    elif args.mode == 'inference':
        print('inference starting...\n')
        trainer.inference_score(config.result_num, args.name+'_inference')
        while 1:
            query = input('English: ')
            if query == 'exit':
                break
            output = trainer.inference(query)
            print('French: ', output, '\n')
        
    else:
        print("Please select mode among 'train', 'test', and 'inference'")
        sys.exit()



if __name__ == '__main__':
    path = os.path.realpath(__file__)
    path = path[:path.rfind('/')+1] + 'config.json'    

    parser = ArgumentParser()
    parser.add_argument('-d', '--device', type=str, required=True, choices=['cpu', 'gpu'])
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test', 'inference'])
    parser.add_argument('-c', '--cont', type=int, default=0, required=False)
    parser.add_argument('-n', '--name', type=str, required=False)
    args = parser.parse_args()

    main(path, args)