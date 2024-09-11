# AmeriOpt
A Python Package for Pricing American Option using Reinforcement Learning.

The full documentation of paper can be found in  [https://www.mdpi.com/1999-4893/17/9/400](https://www.mdpi.com/1999-4893/17/9/400)

![image info](example_mainimage.png)

To use package, you need to follwo the following steps:

## Installation
```bash
pip install ameriopt
```

## Import the package


```python

from ameriopt.rl_policy import RLPolicy
```


## Set the parameters of GBM model

- Number of Laguerre polynomials to be used in the RL model

```python
NUM_LAGUERRE = 5
```

- Number of training iterations for the RL algorithm

```python
TRAINING_ITERS = 3
```

- Small constant for numerical stability in the RL algorithm

```python
EPSILON = 1e-5
```

- Strike price of the option

```python
STRIKE_PRICE = 40
```

- Time to expiration (in years)

```python
EXPIRY_TIME = 1.0
```

- Risk-free interest rate

```python
INTEREST_RATE = 0.06
```

- Number of time intervals 

```python
NUM_INTERVALS = 50
```

- Number of simulations for generating training data

```python
NUM_SIMULATIONS_TRAIN = 5000
```

- Number of simulations for testing the RL policy

```python
NUM_SIMULATIONS_TEST = 10000
```

- Spot price of the underlying asset at the start of the simulation

```python
SPOT_PRICE = 36.0
```

- Volatility of the underlying asset (annualized)

```python
VOLATILITY = 0.2
```


## Simulate Training Data using Geometric Brownian Motion (GBM)


```python
training_data = simulate_GBM_training(
    expiry_time=EXPIRY_TIME,
    num_intervals=NUM_INTERVALS,
    num_simulations=NUM_SIMULATIONS_TRAIN,
    spot_price=SPOT_PRICE,
    interest_rate=INTEREST_RATE,
    volatility=VOLATILITY
)
```

## Instantiate the RLPolicy model with defined parameter GBM Price Model

```python
rl_policy = RLPolicy(
    num_laguerre=NUM_LAGUERRE,
    strike_price=STRIKE_PRICE,
    expiry=EXPIRY_TIME,
    interest_rate=INTEREST_RATE,
    num_steps=NUM_INTERVALS,
    training_iters=TRAINING_ITERS,
    epsilon=EPSILON
)
```

## Train the RL Model and Get Weights (Weight for the optimal policy)


```python
weights = rl_policy.get_weights(training_data=training_data)
```

# Generate test data (GBM paths) for option price scoring

```python
paths_test = scoring_sim_data(
    expiry_time=EXPIRY_TIME,
    num_intervals=NUM_INTERVALS,
    num_simulations_test=NUM_SIMULATIONS_TEST,
    spot_price=SPOT_PRICE,
    interest_rate=INTEREST_RATE,
    volatility=VOLATILITY
)
```

## Option price

```python
option_price = rl_policy.calculate_option_price(stock_paths=paths_test)
```

## Print the calculated option price

```python
print("Option Price using RL Method:", option_price)
```