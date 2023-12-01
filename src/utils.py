import os
import re
import torch
import logging
from datetime import datetime


def save_checkpoint(model, epoch:int, base_dir:str="experiments"):
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
    outdir = os.path.join(base_dir, datetime.today().strftime('%Y-%m-%d'))
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch:.3f}.pt')
    torch.save(model.state_dict(), cpfile)


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