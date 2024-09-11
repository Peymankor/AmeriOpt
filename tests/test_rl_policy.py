# Import necessary modules from the ameriopt package
from ameriopt.rl_policy import RLPolicy
from ameriopt.gbm import simulate_GBM_training
from ameriopt.gbm import scoring_sim_data   

# ---- Set up the Parameters ----
# Number of Laguerre polynomials to be used in the RL model
NUM_LAGUERRE = 5

# Number of training iterations for the RL algorithm
TRAINING_ITERS = 3

# Small constant for numerical stability in the RL algorithm
EPSILON = 1e-5

# Strike price of the option
STRIKE_PRICE = 40

# Time to expiration (in years)
EXPIRY_TIME = 1.0

# Risk-free interest rate
INTEREST_RATE = 0.06

# Number of time intervals (discretization of the time to expiry)
NUM_INTERVALS = 50

# Number of simulations for generating training data
NUM_SIMULATIONS_TRAIN = 5000

# Number of simulations for testing/validation
NUM_SIMULATIONS_TEST = 10000

# Spot price of the underlying asset at the start of the simulation
SPOT_PRICE = 36.0

# Volatility of the underlying asset (annualized)
VOLATILITY = 0.2

# ---- Simulate Training Data using Geometric Brownian Motion (GBM) ----
# Generate training data for the RL model by simulating GBM paths
training_data = simulate_GBM_training(
    expiry_time=EXPIRY_TIME,
    num_intervals=NUM_INTERVALS,
    num_simulations=NUM_SIMULATIONS_TRAIN,
    spot_price=SPOT_PRICE,
    interest_rate=INTEREST_RATE,
    volatility=VOLATILITY
)

# ---- Simulate Testing Data for Scoring ----
# Generate test data (GBM paths) for option price scoring
paths_test = scoring_sim_data(
    expiry_time=EXPIRY_TIME,
    num_intervals=NUM_INTERVALS,
    num_simulations_test=NUM_SIMULATIONS_TEST,
    spot_price=SPOT_PRICE,
    interest_rate=INTEREST_RATE,
    volatility=VOLATILITY
)

# ---- Initialize Reinforcement Learning Policy (RLPolicy) ----
# Instantiate the RLPolicy model with defined parameters
rl_policy = RLPolicy(
    num_laguerre=NUM_LAGUERRE,
    strike_price=STRIKE_PRICE,
    expiry=EXPIRY_TIME,
    interest_rate=INTEREST_RATE,
    num_steps=NUM_INTERVALS,
    training_iters=TRAINING_ITERS,
    epsilon=EPSILON
)

# ---- Train the RL Model and Get Weights ----
# Obtain the optimal policy weights by training the model on the GBM training data
weights = rl_policy.get_weights(training_data=training_data)

# ---- Calculate Option Price using the RL Model ----
# Use the trained RL model to calculate the option price based on the test data
option_price = rl_policy.calculate_option_price(stock_paths=paths_test)

# Print the calculated option price
print("Option Price using RL Method:", option_price)
