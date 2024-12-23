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

"""Model training"""
pinn_model = NeuralNetwork().to(DEVICE)
# pinn_model.load(NeuralNetwork.MODEL_NAME)

t = torch.linspace(*BOUNDS, DOTS, requires_grad=True).view(-1, 1)

optimiser = optim.Adam(pinn_model.parameters(), lr=0.001)
scheduler = StepLR(optimiser, step_size=16, gamma=0.7)
history = pinn_model.pinn_training(optimiser,
                                   scheduler,
                                   duffing.harmonic_oscillator(t.to(DEVICE)),
                                   duffing.solve_numerically(t),
                                   BATCH_SIZE,
                                   DEVICE)
pinn_model.save(NeuralNetwork.MODEL_NAME)

pinn_model = pinn_model.to("cpu")

"""Plotting"""
plotting.start_plotting(pinn_model, history)
