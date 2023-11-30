import json
import os
import torch
from model import ViT
import logging
import re


def save_checkpoint(experiment_name:str, model, epoch:int, base_dir:str="experiments"):
    """
    Saves the model checkpoint

    Args:
    experiment_name (str): Name of the experiment
    model (ViT): The model used for the experiment
    epoch (int): Epoch number
    base_dir (str, optional): Base directory where the experiment is saved. Defaults to "experiments".
    
    Returns:
    None
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)


def load_experiment(experiment_name:str, checkpoint_name:str="model_final.pt", base_dir:str="experiments"):
    """
    Loads the experiment from the saved checkpoint

    Args:
    experiment_name (str): Name of the experiment
    checkpoint_name (str, optional): Name of the checkpoint. Defaults to "model_final.pt".
    base_dir (str, optional): Base directory where the experiment is saved. Defaults to "experiments".

    Returns:
    config (dict): Dictionary containing the config of the experiment
    model (ViT): The model used for the experiment
    train_losses (list): List containing the training losses
    test_losses (list): List containing the test losses
    accuracies (list): List containing the accuracies
    """
    outdir = os.path.join(base_dir, experiment_name)
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    model = ViT(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def lastest_checkpoint(model_dir="./experiments"):
    files = [f for f in os.listdir(model_dir) if 'model' in f]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    if len(files) == 0:
        return None
    return os.path.join(model_dir, files[-1])