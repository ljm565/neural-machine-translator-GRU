# Data Preparation
Here, we will proceed with a GRU translator model training tutorial using the [Tatoeba Project](https://www.manythings.org/anki/) dataset by default.
Please refer to the following instructions to utilize custom datasets.

### 1. Tatoeba
If you want to train on the tatoeba dataset, simply set the `tatoeba_train` value in the `config/config.yaml` file to `True` as follows.
```yaml
tatoeba_train: True       
tatoeba:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
If you want to train your custom dataset, set the `tatoeba_train` value in the `config/config.yaml` file to `False` as follows.
You may require to implement your custom dataloader codes in `src/utils/data_utils.py`.
```yaml
tatoeba_train: False       
tatoeba:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```