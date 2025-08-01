import torch
from torch.utils.data import TensorDataset, DataLoader
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


class TwoFactorLinearRegression:
    def __init__(self, num_features, sigma=0.01, lr=0.01):
        self.num_features = num_features
        self.w = torch.normal(0, sigma, (num_features, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.params = [self.w, self.b]
        self.lr = lr

    def forward(self, X):
        return X @ self.w + self.b

    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return l.mean()

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad.data

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


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
                y_hat = self.model.forward(*inputs)
                loss = self.model.loss(y_hat, y)
                batch_losses.append(loss.item())
                self.model.zero_grad()
                loss.backward()
                self.model.step()

            avg_loss = sum(batch_losses) / len(batch_losses)
            losses.append(avg_loss)

            # === Validation step ===
            batch_val_losses = []
            for batch in self.val_data:
                *inputs, y = batch
                with torch.no_grad():  # Don't update weights
                    y_hat = self.model.forward(*inputs)
                    loss = self.model.loss(y_hat, y)
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

        print(self.model.w.flatten().tolist(), self.model.b.item())
        plt.ioff()
        plt.show()


data = DataGenerator(torch.tensor([2.3, 3.4]), 3.4, 1000, noise=0.01)
model = TwoFactorLinearRegression(2, lr=0.02)
test = Trainer(data, model, 10, num_train=500, batch_size=32)
test.train()
