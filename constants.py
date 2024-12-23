from math import pi as PI


# Training constants
DOTS: int = 10000

BATCH_SIZE: int = 128
EPOCHS: int = 70

LAMBDA: float = 0.001

# Duffing equation parameters
PARAMETERS: (float, float, float, float, float) = 1.0, 0.0, 0.0, 0.0, 1.2
ALPHA, BETA, GAMMA, DELTA, OMEGA = PARAMETERS

T = 2.0 * PI / OMEGA

# Cauchy data
T0: float = 0.0

INITIAL_VALUES: (float, float) = 1.0, 0.0
X0, V0 = INITIAL_VALUES

BOUNDS = (0.0, 13.0)
