import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, w, b, num_total, noise=0.1):
        w = w.reshape(-1, 1)
        noise = torch.randn(num_total, 1) * noise
        self.X = torch.randn(num_total, len(w))
        self.y = self.X @ w + noise + b


class DataHandler:
    def __init__(self, data, train, num_train, batch_size=32):
        indices = slice(0, num_train) if train else slice(num_train, None)
        self.data = tuple(a[indices] for a in data)
        self.batch_size = batch_size
        self.num_train = num_train
        self.dataset = TensorDataset(*self.data)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def __iter__(self):
        return iter(self.loader)


class LinearRegression:
    def __init__(self, lr=0.01):
        self.net = nn.Linear(2, 1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)


class Trainer:
    def __init__(self, data, model, epoch_num, batch_size=32, num_train=100):
        self.model = model
        self.train_data = DataHandler([data.X, data.y], True, num_train, batch_size)
        self.val_data = DataHandler([data.X, data.y], False, num_train, batch_size)
        self.epoch_num = epoch_num

    def train(self):
        losses = []
        val_losses = []

        # Start interactive plotting
        plt.ion()
        # plt.figure()

        for epoch in range(self.epoch_num):
            batch_losses = []
            for batch in self.train_data:
                *inputs, y = batch
                loss = self.model.loss(self.model.net(*inputs), y)
                batch_losses.append(loss.item())
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

            avg_loss = sum(batch_losses) / len(batch_losses)
            losses.append(avg_loss)

            # === Validation step ===
            batch_val_losses = []
            for batch in self.val_data:
                *inputs, y = batch
                with torch.no_grad():  # Don't update weights
                    loss = self.model.loss(self.model.net(*inputs), y)
                    batch_val_losses.append(loss.item())
            avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
            if epoch > 0:
                val_losses.append(avg_val_loss)

            print(
                f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # === Plot both training and validation loss ===
            plt.clf()
            plt.plot(range(1, epoch + 2), losses, "-o", label="Train Loss")
            plt.plot(range(2, epoch + 2), val_losses, "--x", label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.pause(0.1)

        print(
            f"[{', '.join(f'{num:.3f}' for num in self.model.net.weight.flatten().tolist())}], {self.model.net.bias.item():.3f}"
        )
        plt.ioff()
        plt.show()


test = LinearRegression(lr=0.02)
data = DataGenerator(torch.tensor([17.4, 6.9]), 1.2, 1000)
trainer = Trainer(data, test, 12, num_train=500)
trainer.train()
