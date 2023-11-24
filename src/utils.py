import json
import os
import torch
from model import ViT


def save_experiment(experiment_name:str, config:dict, model, train_losses:list, test_losses:list, accuracies:list, base_dir:str="experiments"):
    """
    Saves the experiment after training

    Args:
    experiment_name (str): Name of the experiment
    config (dict): Dictionary containing the config of the experiment
    model (ViT): The model used for the experiment
    train_losses (list): List containing the training losses
    test_losses (list): List containing the test losses
    accuracies (list): List containing the accuracies
    base_dir (str, optional): Base directory where the experiment is saved. Defaults to "experiments".

    Returns:
    None
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


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