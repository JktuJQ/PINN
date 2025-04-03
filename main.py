import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model import NeuralNetwork
import duffing
import plotting
import cuda
from constants import *

"""Torch setup"""
torch.manual_seed(123)

DEVICE = cuda.initialize_device()

"""Numerical methods"""
print()
t = torch.linspace(*BOUNDS, DOTS, requires_grad=True).view(-1, 1)

numerical_solution = duffing.solve_numerically(t)
true_solution = duffing.harmonic_oscillator(t.to(DEVICE))
print(f"MSE of numerical solution with DOTS = {DOTS}:",
      min((true_solution - torch.from_numpy(numerical_solution.y[0]).view(-1, 1).to(DEVICE)) ** 2).item())
print(f"Max Error of numerical solution with DOTS = {DOTS}:",
      max(torch.abs(true_solution - torch.from_numpy(numerical_solution.y[0]).view(-1, 1).to(DEVICE))).item())

"""Model training"""
print()
pinn_model = NeuralNetwork().to(DEVICE)
# pinn_model.load(NeuralNetwork.MODEL_NAME)

optimiser = optim.Adam(pinn_model.parameters(), lr=0.001)
scheduler = StepLR(optimiser, step_size=10, gamma=0.6)
history = pinn_model.pinn_training(optimiser,
                                   scheduler,
                                   true_solution,
                                   numerical_solution,
                                   BATCH_SIZE,
                                   DEVICE)
pinn_model.save(NeuralNetwork.MODEL_NAME)

"""Plotting"""
print()
plotting.start_plotting(pinn_model.to("cpu"), history)
