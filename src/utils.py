import os
import re
import torch
import logging
from datetime import datetime


def save_checkpoint(model, epoch:int, outdir:str):
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
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch:3d}.pt')
    torch.save(model.state_dict(), cpfile)


def set_logger(log_path):
    """
    Sets the logger to log info in terminal and file `log_path`.

    Args:
    log_path (str): Path to the log file.

    Returns:
    None
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
    """
    Get the latest checkpoint file from a directory

    Args:
    model_dir (str): Path to the directory containing the checkpoints

    Returns:
    checkpoint (str): Path to the latest checkpoint file
    """
    files = [f for f in os.listdir(model_dir) if 'model' in f]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    if len(files) == 0:
        return None
    return os.path.join(model_dir, files[-1])