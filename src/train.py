import torch
import torch.nn as nn
import torch.optim as optim
from model import ViT
from utils import save_checkpoint, load_experiment, set_logger, lastest_checkpoint
from data_preprocessing import prepare_data
import os
import matplotlib.pyplot as plt
import logging

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

    def train(self, trainloader, valloader, testloader, epochs: int, early_stop_patience: int = 5):
        """
        Training function for the model

        Args:
        trainloader (DataLoader): DataLoader for training data
        valloader (DataLoader): DataLoader for validation data
        testloader (DataLoader): DataLoader for testing data
        epochs (int): Number of epochs to train the model
        early_stop_patience (int, optional): Number of epochs to wait for improvement in validation loss before early stopping. Defaults to 5.

        Returns:
        None
        """
        train_losses, val_losses, test_losses, accuracies = [], [], [], []
        best_val_loss = float('inf')
        patience_counter = 0

        try:
            for epoch in range(epochs):
                train_loss = self.train_epoch(trainloader)
                val_accuracy, val_loss = self.test(valloader)
                test_accuracy, test_loss = self.test(testloader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                test_losses.append(test_loss)
                accuracies.append(val_accuracy)

                print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}, Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
                logging.info(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}, Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(self.exp_name, self.model, epoch + 1)
                    logging.info("-------- Saved Best Model! --------")
                else:
                    patience_counter += 1
                    logging.info("Early Stop Left: {}".format(patience_counter))

                if early_stop_patience == 0:
                    print(f"Early stopping after {epoch + 1} epochs.")
                    logging.info("-------- Early Stop! --------")
                    break

                if epoch % 10 == 0:
                    save_checkpoint(self.exp_name, self.model, epoch + 1)

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Saving the model...")
            save_checkpoint(self.exp_name, self.model, epoch + 1)
            print("Model saved successfully.")

        return train_losses, val_losses, test_losses, accuracies

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
    
    def continue_train(self, trainloader, valloader, testloader, epochs:int, early_stop_patience:int, checkpoint_name:str):
        """
        Continue training from a checkpoint

        Args:
        trainloader (DataLoader): DataLoader for training data
        testloader (DataLoader): DataLoader for testing data
        epochs (int): Number of epochs to train the model
        early_stop_patience (int): Number of epochs to wait for improvement in validation loss before early stopping
        checkpoint_name (str): Name of the checkpoint file

        Returns:
        None
        """
        try:
            _, _, train_losses, test_losses, accuracies = load_experiment(self.exp_name, checkpoint_name)
            train_losses, val_losses, test_losses, accuracies = [], [], [], []
            for i in range(epochs):
                for epoch in range(epochs):
                    train_loss = self.train_epoch(trainloader)
                    val_accuracy, val_loss = self.test(valloader)
                    test_accuracy, test_loss = self.test(testloader)

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    test_losses.append(test_loss)
                    accuracies.append(val_accuracy)

                    print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}, Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

                    # Check for early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stop_patience:
                        print(f"Early stopping after {epoch + 1} epochs.")
                        break

                    if epoch % 10 == 0:
                        save_checkpoint(self.exp_name, self.model, epoch + 1)

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Saving the model...")
            save_checkpoint(self.exp_name, self.model, i+1)
            print("Model saved successfully.")
        
        return train_losses, val_losses, test_losses, accuracies


def plot_metrics(train_losses, val_losses, accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main(continue_train:bool=False):
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
    early_stop_patience = 5
    log_path = os.path.join("experiments", exp_name, "train.log")

    set_logger(log_path)
    trainloader, valloader, testloader, classes = prepare_data(data_dir, batch_size=batch_size, num_workers=num_workers)
    logging.info("-------- Dataset Build! --------")

    model = ViT(image_size, hidden_size, num_hidden_layers, intermediate_size, len(classes), num_attention_heads, hidden_dropout_prob, 
                attention_probs_dropout_prob, initializer_range, num_channels, patch_size, qkv_bias)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    loss_func = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_func, exp_name, device)

    logging.info("-------- Start Training! --------")
    if continue_train is False:
        train_losses, val_losses, test_losses, accuracies = trainer.train(trainloader, valloader, testloader, epochs, early_stop_patience)
    else:
        model_path = lastest_checkpoint()
        model.load_state_dict(torch.load(model_path))
        logging.info(f"-------- Load Model from {model_path}! --------")
        train_losses, val_losses, test_losses, accuracies = trainer.continue_train(trainloader, valloader, testloader, epochs, early_stop_patience, model)
    logging.info("-------- Training Finished! --------")

    plot_metrics(train_losses, val_losses, accuracies)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()