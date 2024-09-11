from ameriopt.lsm_policy import LSMClass
from ameriopt.gbm import SimulateGBM

# Set up the parameters
num_laguerre_val = 5
training_iters_val = 3
epsilon_val = 1e-5
strike_val = 40
expiry_time_val = 1.0
interest_rate_val = 0.06
num_intervals_val = 50

train_data =SimulateGBM(
    S0=36,
    r=0.06,
    sd=0.2,
    T=1.0,
    paths=100000,
    steps=50,
    reduce_variance=True,
    seed_random=True
)
LSMpolicy= LSMClass(
    spot_price=36,
    strike=40,
    expiry_time=1.0,
    num_intervals=50,
    num_simulations=100000,
    interest_rate=0.06,
    volatility=0.2,
    k=4
)

weights = LSMpolicy.get_policy_lsm(
    train_data)

def payoff_func(_: float, s: float) -> float:
            return max(strike_val - s, 0.)




test_data_v = SimulateGBM(S0=36, r=0.06, 
                          sd=0.2, 
                        T=expiry_time_val, 
                        paths=10000, 
                        steps=50)

option_price = LSMpolicy.option_price_LSM(
    scoring_data=test_data_v,
    Weights_LSM=weights,
    k=4,
    num_intervals=50,
    expiry=1.0,
    interest_rate=0.06,
    payoff_func=payoff_func
)
print("Option Price using LSM Method:", option_price)