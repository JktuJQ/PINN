import torch
import numpy as np
import matplotlib.pyplot as plt

from model import NeuralNetwork
import duffing
from constants import *


def pinn_vs_numerical_plot(t, x, numerical_solution):
    """Plots PINN against numerical methods"""
    plt.plot(t.detach(), x.detach(), color="red", label="PINN")
    plt.plot(np.array(numerical_solution.t), numerical_solution.y[0], color="blue", label="Numerical method")

    plt.xlabel('Time t')
    plt.ylabel('Displacement x(t)')
    plt.title('PINN Solution for Duffing Equation')
    plt.legend()


def phase_diagram_plot(x, v):
    """Plots phase diagram"""
    plt.plot(x.detach(), v.detach())

    plt.xlabel('Displacement x(t)')
    plt.ylabel('Velocity v(t)')
    plt.title('Phase Diagram')


def metrics_plot(history: dict[str, list]):
    """Plots metrics using loss history obtained after training"""

    ax = plt.gca()
    ax.set_yscale('log')

    plt.plot(list(range(1, len(history["PINN"]) + 1)), history["PINN"], label="PINN MSE Loss")
    plt.plot(list(range(1, len(history["NUMERICAL RELATED"]) + 1)), history["NUMERICAL RELATED"],
             label="Numerical Related MSE Loss")
    plt.plot(list(range(1, len(history["MSE"]) + 1)), history["MSE"], label="MSE")
    plt.plot(list(range(1, len(history["MAX ERROR"]) + 1)), history["MAX ERROR"], label="Max Error")

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title("Loss history")
    plt.legend()


def start_plotting(pinn_model: NeuralNetwork, history: dict[str, list]):
    """Starts plotting data"""
    t = torch.linspace(*BOUNDS, 500).view(-1, 1).requires_grad_(True)

    pinn_model.eval()
    pinn_model = pinn_model.to('cpu')
    x = pinn_model(t)

    v = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    x = x.view(-1, 1)
    numerical_solution = duffing.solve_numerically(t)

    plt.figure(figsize=(10, 14))

    plt.subplot(3, 1, 1)
    pinn_vs_numerical_plot(t, x, numerical_solution)

    plt.subplot(3, 1, 2)
    phase_diagram_plot(x, v)

    plt.subplot(3, 1, 3)
    metrics_plot(history)

    plt.show()
