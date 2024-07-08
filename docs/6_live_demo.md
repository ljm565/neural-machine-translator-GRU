# Live Demo
Here, we provide a guide for executing live translator demo with your trained model in a terminal.

### 1. Translator Demo
#### 1.1 Arguments
There are several arguments for running `src/run/inference.py`:
* [`-r`, `--resume_model_dir`]: Directory to the model to execute. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to visualize.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's metric such as BLEU, NIST, etc.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.


#### 1.2 Command
`src/run/inference.py` file is used to execute live demo of your trained translator:
```bash
python3 src/run/inference.py --resume_model_dir ${project}/${name}
```