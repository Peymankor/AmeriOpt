from ameriopt.rl_policy import RLPolicy
from ameriopt.gbm import simulate_GBM_training
from ameriopt.gbm import scoring_sim_data   

# Set up the parameters
num_laguerre_val = 5
training_iters_val = 3
epsilon_val = 1e-5
strike_val = 40
expiry_time_val = 1.0
interest_rate_val = 0.06
num_intervals_val = 50

train_data = simulate_GBM_training(
    expiry_time=expiry_time_val,
    num_intervals=num_intervals_val,
    num_simulations=5000,
    spot_price=36,
    interest_rate=interest_rate_val,
    volatility=0.2
)


num_simulations_test = 10000

paths_test = scoring_sim_data(
    expiry_time=expiry_time_val,
    num_intervals=num_intervals_val,
    num_simulations_test=num_simulations_test,
    spot_price=36.0,
    interest_rate=interest_rate_val,
    volatility=0.2
)

# Initialize RLPolicy
rl_policy = RLPolicy(
    num_laguerre=num_laguerre_val,
    strike_price=strike_val,
    expiry=expiry_time_val,
    interest_rate=interest_rate_val,
    num_steps=num_intervals_val,
    training_iters=training_iters_val,
    epsilon=epsilon_val
)

weights = rl_policy.get_weights(training_data=train_data)

# # Calculate the option price using RLPolicy
option_price = rl_policy.calculate_option_price(stock_paths=paths_test)
print("Option Price using RL Method:", option_price)