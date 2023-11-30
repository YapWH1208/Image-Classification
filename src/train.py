import torch
import torch.nn as nn
import torch.optim as optim
from model import ViT
from utils import save_checkpoint, load_experiment
from data_preprocessing import prepare_data
import os

class Trainer:
    """
    Trainer class for training the model

    Args:
    model (ViT): The model used for the experiment
    optimizer (optim): Optimizer used for training
    loss_func (nn): Loss function used for training
    exp_name (str): Name of the experiment
    device (str): Device to use for training
    """
    def __init__(self, model, optimizer, loss_func:str, exp_name:str, device:str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs:int):
        """
        Training function for the model

        Args:
        trainloader (DataLoader): DataLoader for training data
        testloader (DataLoader): DataLoader for testing data
        epochs (int): Number of epochs to train the model

        Returns:
        None
        """
        train_losses, test_losses, accuracies = [], [], []
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.test(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if i % 10 == 0:
                save_checkpoint(self.exp_name, self.model, i+1)

    def train_epoch(self, trainloader):
        """
        Training function for one epoch

        Args:
        trainloader (DataLoader): DataLoader for training data

        Returns:
        train_loss (float): Training loss
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)
    
    @torch.no_grad()
    def test(self, testloader):
        """
        Testing function for the model

        Args:
        testloader (DataLoader): DataLoader for testing data

        Returns:
        None
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                logits = self.model(images)

                loss = self.loss_func(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss
    
    def continue_train(self, trainloader, testloader, epochs:int, checkpoint_name:str):
        """
        Continue training from a checkpoint

        Args:
        trainloader (DataLoader): DataLoader for training data
        testloader (DataLoader): DataLoader for testing data
        epochs (int): Number of epochs to train the model
        checkpoint_name (str): Name of the checkpoint file

        Returns:
        None
        """
        _, _, train_losses, test_losses, accuracies = load_experiment(self.exp_name, checkpoint_name)
        train_losses, test_losses, accuracies = [], [], []
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.test(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if i % 10 == 0:
                save_checkpoint(self.exp_name, self.model, i+1)

def main():
    # Hyperparameters
    batch_size = 128
    epochs = 100
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = "ViT"
    num_workers = 2
    data_dir = "./data/"
    patch_size = 16
    hidden_size = 48
    num_hidden_layers = 4
    num_attention_heads = 4
    intermediate_size = 4 * hidden_size
    hidden_dropout_prob = 0.0
    attention_probs_dropout_prob = 0.0
    initializer_range = 0.02
    image_size = 224
    num_channels = 3
    qkv_bias = True

    trainloader, testloader, classes = prepare_data(data_dir, batch_size=batch_size, num_workers=num_workers)
    #print(classes)

    model = ViT(image_size, hidden_size, num_hidden_layers, intermediate_size, len(classes), num_attention_heads, hidden_dropout_prob, 
                attention_probs_dropout_prob, initializer_range, num_channels, patch_size, qkv_bias)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    loss_func = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_func, exp_name, device)
    trainer.train(trainloader, testloader, epochs)
    trainer.continue_train(trainloader, testloader, epochs, "model_10.pt")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()