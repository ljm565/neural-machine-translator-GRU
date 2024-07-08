# Neural Machine Translator GRU
한국어 버전의 설명은 [여기](./docs/README_ko.md)를 참고하시기 바랍니다.

## Introduction
We will create a GRU-based machine translation model using the English-French sentence pair data from the Tatoeba Project. This code will allow you to create a machine translation model with or without the use of attention and [scheduled sampling](https://arxiv.org/pdf/1506.03099.pdf).
For an explanation of the GRU-based machine translation model and its attention mechanism, please refer to [Sequence-to-Sequence (Seq2Seq) 모델과 Attention](https://ljm565.github.io/contents/RNN2.html).
Furthermore, the attention in this code is implemented based on [Bahdanau Attention](https://arxiv.org/pdf/1409.0473.pdf) and is not related to the [PyTorch seq2seq tutorial](https://tutorials.pytorch.kr/intermediate/seq2seq_translation_tutorial.html).
If you want to see a model using a different attention mechanism, please refer to the code in the [PyTorch seq2seq tutorial](https://tutorials.pytorch.kr/intermediate/seq2seq_translation_tutorial.html).
<br><br><br>

## Supported Model
### Seqeunce-to-Sequence GRU Model
* English-French 기계 번역 모델 제작을 위해 GRU 기반의 seq2seq 모델을 학습합니다.
* A GRU using `nn.GRU` is implemented.
* Bahdanau Attention (You can decide in `config/config.yaml` whether to use attention or not).
<br><br><br>


## Supported Tokenizer
### Custom Word Tokenizer
* Tokenization based on words for attention visualization.
<br><br><br>

## Base Dataset
* Base dataset for tutorial is English-French dataset of [Tatoeba Project](https://www.manythings.org/anki/).
<br><br><br>

## Supported Devices
* CPU, GPU, multi-GPU (DDP), MPS (for Mac and torch>=1.12.0)
<br><br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>


## Project Tree
This repository is structured as follows.
```
├── configs                           <- Folder for storing config files
│   └── *.yaml
│
└── src      
    ├── models
    |   └── gru_seq2seq.py            <- GRU model file
    |
    ├── run                   
    |   ├── inference.py              <- Trained model live demo execution code
    |   ├── train.py                  <- Training execution file
    |   ├── validation.py             <- Trained model evaulation execution file
    |   └── vis_attention.py          <- Attention visualization code for each word of attention model
    |
    ├── tools                   
    |   ├── tokenizers
    |   |    └── word_tokenizer.py    <- Word tokenizer file
    |   |
    |   ├── early_stopper.py          <- Early stopper class file
    |   ├── evaluator.py              <- Metric evaluator class file
    |   ├── model_manager.py          
    |   ├── tatoeba_downloader.py     <- Tatoeba data download file  
    |   └── training_logger.py        <- Training logger class file
    |
    ├── trainer                 
    |   ├── build.py                  <- Codes for initializing dataset, dataloader, etc.
    |   └── trainer.py                <- Class for training, evaluating, and calculating accuracy
    |
    └── uitls                   
        ├── __init__.py               <- File for initializing the logger, versioning, etc.
        ├── data_utils.py             <- File defining the custom dataset dataloader
        ├── filesys_utils.py       
        ├── func_utils.py       
        └── training_utils.py     
```
<br><br>


## Tutorials & Documentations
Please follow the steps below to train a GRU translator model.
1. [Getting Started](./docs/1_getting_started.md)
2. [Data Preparation](./docs/2_data_preparation.md)
3. [Training](./docs/3_trainig.md)
4. ETC
   * [Evaluation](./docs/4_model_evaluation.md)
   * [Attention Visualization](./docs/5_vis_attn.md)
   * [Live Demo](./docs/6_live_demo.md)

<br><br><br>


* ### 모델 학습 조건 설정 (config.json)
    **주의사항: 최초 학습 시 config.json이 사용되며, 이미 한 번 학습을 한 모델에 대하여 parameter를 바꾸고싶다면 base_path/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.**
    * is_attn: {0, 1} 중 선택. Attention 모델을 제작한다면 1, 아니면 0. 
    * base_path: 학습 관련 파일이 저장될 위치.
    * model_name: 학습 모델이 저장될 파일 이름 설정. 모델은 base_path/model/{model_name}/{model_name}.pt 로 저장.
    * loss_data_name: 학습 시 발생한 loss data를 저장하기 위한 이름 설정. base_path/loss/{loss_data_name}.pkl 파일로 저장. 내부에 중단된 학습을 다시 시작할 때, 학습 과정에 발생한 loss 데이터를 그릴 때 등 필요한 데이터를 dictionary 형태로 저장.
    * vocab_size: 최대 vocab size 설정.
    * max_len: 토큰화 된 번역 source, target 데이터의 최대 길이.
    * hidden_size: GRU 모델의 hidden dimension.
    * num_layers: GRU 모델의 레이어 수.
    * dropout: 모델의 dropout 비율.
    * batch_size: batch size 지정.
    * epochs: 학습 epoch 설정.
    * lr: learning rate 지정.
    * teacher_forcing_ratio: Teacher forcing (교사 강요) 비율. 1일 경우 모든 학습이 teacher forcing으로 이루어짐(e.g. 0.9일 경우 학습의 step마다 10 %의 확률로 scheduled sampling 방식으로 학습이 진행됨). 
    * result_num: 모델 테스트 시, 결과를 보여주는 sample 개수.
    * early_stop_criterion: Test set의 최소 loss를 내어준 학습 epoch 대비, 설정된 숫자만큼 epoch이 지나도 나아지지 않을 경우 학습 조기 종료.
    * visualize_attn: {0, 1} 중 선택. 1이면 랜덤으로 result_num에서 설정해준 개수만큼 랜덤으로 attention score를 가시화하여 {base_path}/result 폴더에 모델 이름으로 이미지 저장.
    <br><br><br>


## 결과
* ### Neural Machine Translator GRU 모델별 결과
    아래 loss, score의 결과는 inference의 결과가 아닌 teacher forcing으로 확인한 결과입니다.
    그리고 아래 표기된 결과는 test set에서 가장 낮은 loss를 가진 모델의 점수입니다.
    따라서 그래프에서 보이는 학습 중 best score와 차이가 있을 수 있습니다.
    마지막으로 inference 방식으로 계산된 loss 및 score를 보고싶다면 inference mode로 실행 시 자동 계산 되므로 확인할 수 있습니다.
    * Training Set Loss History<br>
        <img src="docs/figs/trainLoss.png" width="80%"><br><br>

    * Test Set Loss History<br>
        <img src="docs/figs/testLoss.png" width="80%"><br>
        * Model with Attention: 0.3367
        * Scheduled Sampling Model with Attention: 0.3509
        * Model without Attention: 0.3366
        * Scheduled Sampling Model without Attention: 0.3491<br><br>

    * Test Set Perplexity (PPL) History<br>
        <img src="docs/figs/testPPL.png" width="80%"><br>
        * Model with Attention: 1.4003
        * Scheduled Sampling Model with Attention: 1.4203
        * Model without Attention: 1.4002
        * Scheduled Sampling Model without Attention: 1.4178<br><br>

    * BLEU-2 Score History<br>
        <img src="docs/figs/bleu2.png" width="80%"><br>
        * Model with Attention: 0.5789
        * Scheduled Sampling Model with Attention: 0.5646
        * Model without Attention: 0.5735
        * Scheduled Sampling Model without Attention: 0.5656<br><br>

    * BLEU-4 Score History<br>
        <img src="docs/figs/bleu4.png" width="80%"><br>
        * Model with Attention: 0.3996
        * Scheduled Sampling Model with Attention: 0.3834
        * Model without Attention: 0.3893
        * Scheduled Sampling Model without Attention: 0.3849<br><br>

    * NIST-2 Score History<br>
        <img src="docs/figs/nist2.png" width="80%"><br>
        * Model with Attention: 6.8475
        * Scheduled Sampling Model with Attention: 6.6922
        * Model without Attention: 6.8016
        * Scheduled Sampling Model without Attention: 6.7098<br><br>

    * NIST-4 Score History<br>
        <img src="docs/figs/nist4.png" width="80%"><br>
        * Model with Attention: 7.1627
        * Scheduled Sampling Model with Attention: 7.0052
        * Model without Attention: 7.1178
        * Scheduled Sampling Model without Attention: 7.0177<br><br>


    * 기계 번역 결과 샘플<br>
        Inference의 결과가 아닌 teacher forcing으로 확인한 결과입니다.
        Inference 방식으로 계산된 번역 결과 및 attention 가시화를 하고싶다면 inference mode로 실행 시 확인할 수 있습니다.
        그리고 inference를 할 때 위의 score 및 결과 샘플을 내어준 후, 실제 번역기 테스트가 가능합니다.
        * Model with Attention
            ```
            # Sample 1
            src : when i was your age , i had a girlfriend .
            gt  : lorsque j'avais votre age , j'avais une petite amie .
            pred: lorsque j'avais votre age , j'avais une petite amie .


            # Sample 2
            src : he gave me some money .
            gt  : il me donna un peu d'argent .
            pred: il me donna un peu d'argent .


            # Sample 3
            src : please answer all the questions .
            gt  : repondez a toutes les questions , s'il vous plait .
            pred: repondez a toutes les questions , s'il vous plait .

            ```
            <img src="docs/figs/nmt_GRU_Attn_attention0.jpg" width="32%">
            <img src="docs/figs/nmt_GRU_Attn_attention1.jpg" width="32%">
            <img src="docs/figs/nmt_GRU_Attn_attention2.jpg" width="32%"><br><br><br>

        * Scheduled Sampling Model with Attention
            ```
            # Sample 1
            src : i'm in love with you and i want to marry you .
            gt  : je suis amoureuse de toi et je veux me marier avec toi .
            pred: je vous amoureux de toi et je veux vous epouser . toi .


            # Sample 2
            src : what's really going one here ?
            gt  : que se passe-t-il vraiment ici ?
            pred: que se passe-t-il, ici ?


            # Sample 3
            src : we do need your advice .
            gt  : il nous faut ecouter vos conseils .
            pred: nous nous faut que tes conseils .

            ```
            <img src="docs/figs/nmt_GRU_Attn_ss_attention0.jpg" width="32%">
            <img src="docs/figs/nmt_GRU_Attn_ss_attention1.jpg" width="32%">
            <img src="docs/figs/nmt_GRU_Attn_ss_attention2.jpg" width="32%"><br><br><br>

        * Model without Attention
            ```
            # Sample 1
            src : tom asked mary for some help .
            gt  : tom a demande a mary de l'aider .
            pred: tom demande demande a mary de l'aide .


            # Sample 2
            src : you see what i mean ?
            gt  : tu vois ce que je veux dire ?
            pred: tu vois ce que je veux dire ?


            # Sample 3
            src : i haven't talked to you in a while .
            gt  : je ne t'ai pas parle depuis un bon moment .
            pred: je n'ai vous pas parle pendant un moment moment .

            ```
            <br><br>

        * Scheduled Sampling Model without Attention
            ```
            # Sample 1
            src : let's take a little break .
            gt  : faisons une petite pause .
            pred: faisons une pause pause .


            # Sample 2
            src : they live on the [UNK] floor of this [UNK] .
            gt  : ils vivent au [UNK] etage de ces [UNK] .
            pred: ils vivent au sujet de de ce sujets .


            # Sample 3
            src : tom doesn't understand why mary is so popular .
            gt  : tom ne comprend pas pourquoi marie est si populaire .
            pred: tom ne comprend pas pourquoi mary est si populaire .

            ```
            <br><br>



<br><br><br>
