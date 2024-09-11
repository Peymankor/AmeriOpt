from typing import List, Tuple
import numpy as np
import math


def simulate_GBM_training(
    expiry_time: float,
    num_intervals: int,
    num_simulations: int,
    spot_price: float,
    interest_rate: float,
    volatility: float
) -> List[Tuple[int, float, float]]:

    """
    Generates simulated stock price data for training purposes.

    Parameters:
        expiry_time (float): expiration date (year).
        num_intervals (int): Number of exercise time steps.
        num_simulations (int): Number of independent price paths to generate.
        spot_price (float): Starting stock price, spot price
        interest_rate (float): Risk-free interest rate.
        volatility (float): Volatility of the stock price.

    Returns:
        List[Tuple[int, float, float]]: List of tuples containing (time step, current price, next price).
    """

    results = []
    time_step_size = expiry_time / num_intervals
    volatility_squared = volatility ** 2

    for _ in range(num_simulations):
        
        current_price = spot_price

        for step in range(num_intervals):
            mean = np.log(current_price) + (interest_rate - 0.5 * volatility_squared) * time_step_size
            variance = volatility_squared * time_step_size
            next_price = np.exp(np.random.normal(mean, np.sqrt(variance)))
            results.append((step, current_price, next_price))
            current_price = next_price

    return results

def scoring_sim_data(
    expiry_time: float,
    num_intervals: int,
    num_simulations_test: int,
    spot_price: float,
    interest_rate: float,
    volatility: float
) -> np.ndarray:
    
    paths: np.ndarray = np.empty([num_simulations_test, num_intervals + 1])
    time_step_size = expiry_time / num_intervals

    #vol2: float = volatility * volatility
    volatility_squared = volatility ** 2

    for i in range(num_simulations_test):
        paths[i, 0] = spot_price
        for step in range(num_intervals):
            mean: float = np.log(paths[i, step]) + (interest_rate - volatility_squared / 2) * time_step_size
            variance = volatility_squared * time_step_size
            paths[i, step + 1] = np.exp(np.random.normal(mean, np.sqrt(variance)))
    return paths[:,1:]

def SimulateGBM(S0, r, sd, T, paths, steps, reduce_variance = True, 
          seed_random = True):
    
    if seed_random:
      np.random.seed(1)

    steps = int(steps)
    dt = T/steps
    Z = np.random.normal(0, 1, paths//2 * steps).reshape((paths//2, steps))
    # Z_inv = np.random.normal(0, 1, paths//2 * steps).reshape((paths//2, steps))
    if reduce_variance:
      Z_inv = -Z
    else:
      Z_inv = np.random.normal(0, 1, paths//2 * steps).reshape((paths//2, steps))
    dWt = math.sqrt(dt) * Z
    dWt_inv = math.sqrt(dt) * Z_inv
    dWt = np.concatenate((dWt, dWt_inv), axis=0)
    St = np.zeros((paths, steps + 1))
    St[:, 0] = S0
    for i in range (1, steps + 1):
        St[:, i] = St[:, i - 1]*np.exp((r - 1/2*np.power(sd, 2))*dt + sd*dWt[:, i-1])
    
    return St[:,1:]
    #return St
