import torch
from scipy.integrate import solve_ivp

from math import sqrt, cos, asin, atan

from constants import *


def differential_equation(t, x, v, a, cos_fn):
    """Return Duffing differential equation"""
    return a + DELTA * v + ALPHA * x + BETA * (x ** 3) - GAMMA * cos_fn(OMEGA * t)


def residual(t: torch.Tensor, x: torch.Tensor):
    """Finds residual from Duffing equation by using `torch.autograd`."""
    v = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    a = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    return differential_equation(t, x, v, a, torch.cos)


def equations_system(t: float, solution: (float, float)):
    """System of differential equations that describes Duffing oscillator"""
    x, v = solution
    dx_dt = v
    dv_dt = -differential_equation(t, x, v, 0.0, cos)
    return dx_dt, dv_dt


def harmonic_oscillator(t):
    """Analytic solution for harmonic oscillator"""
    amplitude = sqrt(X0 ** 2 + V0 ** 2 / ALPHA)
    phi = asin(X0 / amplitude)
    w = sqrt(ALPHA)
    return amplitude * torch.sin(phi + w * t)


def damped_oscillator(t):
    """Analytic solution for damped oscillator"""
    omega = sqrt(ALPHA - DELTA ** 2 / 4.0)
    phi = -atan(DELTA / 2.0 * omega)
    amplitude = 1.0 / cos(phi)
    return amplitude * torch.exp(-DELTA / 2.0 * t) * torch.cos(omega * t + phi)


def solve_numerically(t):
    """Solves Duffing equation numerically. **This requires that `t` must be detached from GPU!**"""
    return solve_ivp(equations_system, BOUNDS, (X0, V0), t_eval=t.detach().view(1, -1).numpy()[0])
