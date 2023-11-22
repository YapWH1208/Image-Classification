import torch
import torch.nn as nn
import torch.optim as optim
from model import ViT
from utils import save_checkpoint, save_experiment
from data_preprocessing import prepare_data

config = {
    "patch_size": 16,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, 
    "num_channels": 3,
    "qkv_bias": True,
}

class Trainer:
    def __init__(self, model, optimizer, loss_func, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs):
        train_losses, test_losses, accuracies = [], [], []
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if i % 10 == 0:
                save_checkpoint(self.exp_name, self.model, i+1)
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            self.optimizer.zero_grad()
            outputs = self.model(batch[0])
            loss = self.loss_func(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)
    
    @torch.no_grad()
    def test(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                logits = self.model(images)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss

def main():
    batch_size = 128
    epochs = 100
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = "ViT"
    
    trainloader, testloader = prepare_data(batch_size=batch_size)

    model = ViT(config)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    loss_func = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_func, exp_name, device)
    trainer.train(trainloader, testloader, epochs)

if __name__ == "__main__":
    main()