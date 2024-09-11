import numpy as np
from numpy.polynomial.laguerre import lagval
from typing import List, Tuple, Callable

class RLPolicy:
    def __init__(self, num_laguerre: int, strike_price: float, expiry: float, interest_rate: float, num_steps: int, training_iters: int, epsilon: float):
        self.num_laguerre = num_laguerre
        self.strike_price = strike_price
        self.expiry = expiry
        self.interest_rate = interest_rate
        self.num_steps = num_steps
        self.training_iters = training_iters
        self.epsilon = epsilon
        self.weights = None

    def make_feature_functions(self, state: Tuple[float, float]) -> List[Callable[[Tuple[float, float]], float]]:
        ident = np.eye(self.num_laguerre)

        def time_feature(t_s, i, expiry, ident):
            return np.exp(-t_s[0] / (2 * expiry)) * lagval(t_s[0] / expiry, ident[i])

        def price_feature(t_s, i, strike, ident):
            return np.exp(-t_s[1] / (2 * strike)) * lagval(t_s[1] / strike, ident[i])

        features: List[Callable[[Tuple[float, float]], float]] = []

        for i in range(self.num_laguerre):
            features.append(time_feature(state, i, self.expiry, ident))

        for i in range(self.num_laguerre):
            features.append(price_feature(state, i, self.strike_price, ident))

        return features

    def get_weights(self, training_data: List[Tuple[int, float, float]]) -> np.ndarray:
        dt = self.expiry / self.num_steps
        gamma = np.exp(-self.interest_rate * dt)
        num_features = self.num_laguerre * 2

        states = [(i * dt, s) for i, s, _ in training_data]
        next_states = [((i + 1) * dt, s1) for i, _, s1 in training_data]

        feature_vals = np.array([self.make_feature_functions(state) for state in states])
        next_feature_vals = np.array([self.make_feature_functions(next_state) for next_state in next_states])

        non_terminal_flags = np.array([index < self.num_steps - 1 for index, _, _ in training_data])
        exercise_value = np.array([max(self.strike_price - stock_price, 0) for _, stock_price in next_states])

        wts = np.ones(num_features)

        for _ in range(self.training_iters):
            a_inv = np.eye(num_features) / self.epsilon
            b_vec = np.zeros(num_features)
            cont = np.dot(next_feature_vals, wts)
            cont_cond = non_terminal_flags * (cont > exercise_value)

            for i in range(len(training_data)):
                phi1 = feature_vals[i]
                phi2 = phi1 - cont_cond[i] * gamma * next_feature_vals[i]

                # Sherman-Morrison formula for matrix a_inv
                temp = a_inv.T.dot(phi2)
                a_inv -= np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))

                # matrix b update
                b_vec += phi1 * (1 - cont_cond[i]) * exercise_value[i] * gamma

            wts = a_inv.dot(b_vec)

        self.weights = wts
        return wts

    def calculate_option_price(self, stock_paths: np.ndarray) -> float:
        num_paths = stock_paths.shape[0]
        num_time_steps = stock_paths.shape[1]
        option_prices = np.zeros(num_paths)
        dt = self.expiry / num_time_steps

        for path_index, price_path in enumerate(stock_paths):
            time_step = 1
            while time_step <= num_time_steps:
                current_time = time_step * dt
                current_stock_price = price_path[time_step - 1]
                exercise_value = max(self.strike_price - current_stock_price, 0)

                state = (current_time, current_stock_price)
                feature_values = self.make_feature_functions(state)
                continue_price = np.dot(feature_values, self.weights) if time_step < num_time_steps else 0

                time_step += 1
                if (exercise_value >= continue_price) and (exercise_value > 0):
                    option_prices[path_index] = np.exp(-self.interest_rate * current_time) * exercise_value
                    time_step = num_time_steps + 1

        return np.average(option_prices)

# # Example usage:

# # Set up the parameters
# num_laguerre_val = 3
# training_iters_val = 4
# epsilon_val = 1e-5
# strike_val = 40
# expiry_time_val = 1.0
# interest_rate_val = 0.06
# num_intervals_val = 50


# # Initialize RLPolicy
# rl_policy = RLPolicy(
#     num_laguerre=num_laguerre_val,
#     strike_price=strike_val,
#     expiry=expiry_time_val,
#     interest_rate=interest_rate_val,
#     num_steps=num_intervals_val,
#     training_iters=training_iters_val,
#     epsilon=epsilon_val
# )

# # Train the RLPolicy to get weights
# weights = rl_policy.get_weights(training_data=RL_data_training)

# # Calculate the option price using RLPolicy
# option_price = rl_policy.calculate_option_price(stock_paths=paths_test)
# print("Option Price using RL Method:", option_price)
