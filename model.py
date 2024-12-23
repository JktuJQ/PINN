import torch
import torch.nn as nn
import numpy as np

import duffing

from constants import *


class Sin(nn.Module):
    """Implements `sin` activation function"""

    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class NeuralNetwork(nn.Module):
    """Neural network that approximates Duffing equation using PINN method"""

    MODEL_NAME = "PINN"

    def __init__(self, configuration: nn.Sequential = None):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),
            Sin(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            Sin(),
            nn.Linear(32, 1),
        ) if configuration is None else configuration

    def forward(self, t):
        return self.layers(t)

    def pinn_training(self, optimiser, scheduler, numerical_solution: np.array, batch_size: int, device: torch.device) -> dict[str, list]:
        self.train(True)

        T0_TENSOR = torch.tensor([T0], requires_grad=True).to(device)
        T_COLUMN = torch.linspace(*BOUNDS, DOTS, requires_grad=True).view(-1, 1).to(device)

        if batch_size is None:
            batch_size = DOTS
        indices = torch.randperm(DOTS)
        batches = [indices[i:i + batch_size] for i in range(0, DOTS, batch_size)]

        history = {"PINN": [], "NUMERICAL RELATED": []}
        for epoch in range(EPOCHS):
            epoch_pinn_loss = 0.0
            epoch_numerical_loss = 0.0

            for batch_indices in batches:
                optimiser.zero_grad()

                # Get the current batch
                t_batch = T_COLUMN[batch_indices]
                x = self(t_batch).to(device)

                # Numerical loss
                numerical_loss = torch.mean(
                    (x - torch.from_numpy(numerical_solution.y[0][batch_indices]).to(device)) ** 2)

                # Residual loss
                residual = duffing.residual(t_batch, x)
                loss_residual = torch.mean(residual ** 2)

                # Initial conditions
                x0_predicted = self(T0_TENSOR).to(device)
                v0_predicted = torch.autograd.grad(x0_predicted, T0_TENSOR, create_graph=True)[0]
                loss_iv = torch.mean((x0_predicted - X0) ** 2) + torch.mean((v0_predicted - V0) ** 2)

                # Total loss
                loss = loss_residual + LAMBDA * loss_iv
                loss.backward()
                epoch_pinn_loss += loss.item()
                epoch_numerical_loss += numerical_loss.item()
                optimiser.step()

            scheduler.step()
            history["PINN"].append(epoch_pinn_loss / len(batches))
            history["NUMERICAL RELATED"].append(epoch_numerical_loss / len(batches))
            if (epoch + 1) % 16 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}, NRLoss: {numerical_loss.item()}, Learning Rate: {lr}')
        return history

    def save(self, filename: str):
        torch.save(self.state_dict(), f"data/{filename}")

    def load(self, filename: str):
        self.load_state_dict(torch.load(f"data/{filename}", weights_only=True))
