import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def parse_dat(file):
    t = []
    x = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            t += [float(line.split()[0])]
            x += [float(line.split()[1])]
    return (t,x)


print(f"CUDA version: {torch.version.cuda}")
is_cuda = torch.cuda.is_available()
print(f"Is CUDA supported by this system? {is_cuda}")
if is_cuda:
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
device = torch.device('cuda') if is_cuda else torch.device('cpu')
print(f"Было выбрано: {device}")


class Sin(nn.Module):
    '''Implements the sin activation function'''

    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


# Duffing equation parameters
alpha = 1.0
beta = 0.0
gamma = 0.37
delta = 0.3
omega = 1.2
# Initial conditions
x0 = 1.0
v0 = 0.0
t0 = torch.tensor([0.0], requires_grad=True).to(device)


# Neural network definition
class DuffingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(1, 32),
            Sin(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            Sin(),
            nn.Linear(32, 1),
        )

    def forward(self, t):
        t = self.f(t)
        return t


def duffing_residual(t, x, delta, alpha, beta, gamma, omega):
    x_t = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    residual = x_tt + delta * x_t + alpha * x + beta * x ** 3 - gamma * torch.cos(omega * t)
    return residual


def duffing(t, s, alpha, beta, gamma, delta, omega):
    x, v = s
    dxdt = v
    dvdt = gamma * np.cos(omega * t) - delta * v - alpha * x - beta * x ** 3
    return dxdt, dvdt


torch.manual_seed(123)
model = DuffingNet().to(device)
model_name = "oscillator_model1"
# model.load_state_dict(torch.load(f"data/{model_name}", weights_only=True))


T = 2*np.pi / omega
bounds = (0.1, 3*T)
N = 5000
t_col = torch.linspace(*bounds, N, requires_grad=True).view(-1, 1)
sol = solve_ivp(duffing, bounds, (x0, v0), t_eval=t_col.detach().view(1, -1).numpy()[0], args=(alpha, beta, gamma, delta, omega))
t_col = t_col.to(device)

# Create batches
batch_size = 128
indices = torch.randperm(N)
batches = [indices[i:i + batch_size] for i in range(0, N, batch_size)]

# Model and optimizer setup
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = ExponentialLR(optimizer, gamma=0.99)
scheduler = StepLR(optimizer, step_size=16, gamma=0.7)
num_epochs = 256

# Training loop
model.train()
pinn_loss_history = []
numerical_loss_history = []

for epoch in range(num_epochs):
    epoch_pinn_loss = 0.0
    epoch_numerical_loss = 0.0

    for batch_indices in batches:
        optimizer.zero_grad()

        # Get the current batch
        t_batch = t_col[batch_indices]
        x = model(t_batch).to(device)

        # Numerical loss
        numerical_loss = torch.mean((x - torch.from_numpy(sol.y[0][batch_indices]).to(device)) ** 2)

        # Residual loss
        residual = duffing_residual(t_batch, x, delta, alpha, beta, gamma, omega)
        loss_residual = torch.mean(residual ** 2)

        # Initial conditions
        x0_pred = model(t0).to(device)
        v0_pred = torch.autograd.grad(x0_pred, t0, create_graph=True)[0]
        loss_ic = torch.mean((x0_pred - x0) ** 2) + torch.mean((v0_pred - v0) ** 2)

        # Total loss
        loss = 1000 * loss_residual + 0.1 * loss_ic
        loss.backward()
        epoch_pinn_loss += loss.item()
        epoch_numerical_loss += numerical_loss.item()
        optimizer.step()

    scheduler.step()
    pinn_loss_history.append(epoch_pinn_loss / len(batches))
    numerical_loss_history.append(epoch_numerical_loss / len(batches))
    if (epoch + 1) % 16 == 0:
        lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, NRLoss: {numerical_loss.item()}, Learning Rate: {lr}')

torch.save(model.state_dict(), f"data/{model_name}")

model = DuffingNet()
model.load_state_dict(torch.load(f"data/{model_name}", weights_only=True))

# Evaluation and plotting
model.eval()
t_test = torch.linspace(*bounds,500).view(-1,1).requires_grad_(True)
x = model(t_test)
residual_test = duffing_residual(t_test, x, delta, alpha, beta, gamma, omega).detach().numpy()
dx  = torch.autograd.grad(x, t_test, torch.ones_like(x), create_graph=True)[0].detach()
x = x.detach().view(-1, 1)
sol = solve_ivp(duffing, bounds, (x0, v0), t_eval=t_test.detach().view(1, -1).numpy()[0], args=(alpha, beta, gamma, delta, omega))

plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
plt.plot(t_test.detach().numpy(), x, color="red", label="PINN")
plt.plot(np.array(sol.t), sol.y[0], color="blue", label="Numerical method")
plt.xlabel('Time t')
plt.ylabel('Displacement x(t)')
plt.title('PINN Solution for Duffing Equation')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_test.detach().numpy(), residual_test)
plt.xlabel('Time t')
plt.ylabel('Residual')
plt.title('Residual of Duffing Equation')

plt.subplot(2, 2, 3)
plt.plot(x, dx)
plt.xlabel('Displacement x(t)')
plt.ylabel('Velocity v(t)')
# plt.title('Phase Diagram')

# plt.subplot(2, 2, 4)
# plt.plot(t_test.detach().numpy(), dx)
# plt.xlabel('t')
# plt.ylabel('Velocity v(t)')

plt.subplot(2, 2, 4)
ax=plt.gca()
ax.set_yscale('log')
plt.plot(list(range(1, len(pinn_loss_history) + 1)), pinn_loss_history, label="PINN MSE Loss")
plt.plot(list(range(1, len(pinn_loss_history) + 1)), numerical_loss_history, label="Numerical Related MSE Loss")
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title("Loss history")
plt.legend()

plt.show()