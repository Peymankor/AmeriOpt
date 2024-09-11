import numpy as np
from typing import Callable, List, Tuple
from ameriopt.helper import laguerre_polynomials, laguerre_polynomials_ind

class LSMClass:
    def __init__(self, spot_price, strike, expiry_time, num_intervals, num_simulations, interest_rate, volatility, k):
        self.spot_price = spot_price
        self.strike = strike
        self.expiry_time = expiry_time
        self.num_intervals = num_intervals
        self.num_simulations = num_simulations
        self.interest_rate = interest_rate
        self.volatility = volatility
        self.k = k

    def get_policy_lsm(self, training_data):
        steps = int(self.num_intervals)
        Stn = training_data

        dt = self.expiry_time / steps
        cashFlow = np.zeros((self.num_simulations, steps))
        cashFlow[:, steps - 1] = np.maximum(self.strike - Stn[:, steps - 1], 0)

        cont_value = cashFlow

        decision = np.zeros((self.num_simulations, steps))
        decision[:, steps - 1] = 1
        Weights_LSM = {}

        for index, i in enumerate(reversed(range(steps - 1))):
            # Find in the money paths
            in_the_money_n = np.where(self.strike - Stn[:, i] > 0)[0]
            out_of_money_n = np.asarray(list(set(np.arange(self.num_simulations)) - set(in_the_money_n)))

            X = laguerre_polynomials(Stn[in_the_money_n, i], self.k)
            Y = cashFlow[in_the_money_n, i + 1] / np.exp(self.interest_rate * dt)

            A = np.dot(X.T, X)
            b = np.dot(X.T, Y)
            Beta = np.dot(np.linalg.pinv(A), b)

            cont_value[in_the_money_n, i] = np.dot(X, Beta)
            try:
                cont_value[out_of_money_n, i] = cont_value[out_of_money_n, i + 1] / np.exp(
                    self.interest_rate * dt)
            except:
                pass

            decision[:, i] = np.where(np.maximum(self.strike - Stn[:, i], 0) - cont_value[:, i] >= 0, 1, 0)
            cashFlow[:, i] = np.maximum(self.strike - Stn[:, i], cont_value[:, i])

            Weights_LSM.update({i: Beta})

        return Weights_LSM
    
    def option_price_LSM(self, scoring_data: np.ndarray,
        Weights_LSM,
        k,
        num_intervals,
        expiry,
        interest_rate,
        payoff_func: Callable[[float, float], float]
        #func: FunctionApprox[Tuple[float, float]]
    ) -> float:

        num_steps = num_intervals-1
        num_paths: int = scoring_data.shape[0]
        prices: np.ndarray = np.zeros(num_paths)
        #stoptime: np.ndarray = np.zeros(num_paths)
        dt: float =  expiry/ num_intervals

        #Beta_list.reverse()
        
        for i, path in enumerate(scoring_data):
            step: int = 0
            while step <= num_steps:
                t: float = (step+1) * dt
                exercise_price: float = payoff_func(t, path[step])

                if exercise_price>0:
                    XX=laguerre_polynomials_ind(path[step],k)
                    continue_price: float = np.dot(XX, Weights_LSM[step])  \
                        if step < num_steps else 0.

                #continue_price: float = func.evaluate([(t, path[step])])[0] \
                #    if step < self.num_steps else 0.
                    step += 1
                    if exercise_price >= continue_price:
                        prices[i] = np.exp(-interest_rate * t) * exercise_price
                        #stoptime[i] = step
                        step = num_steps + 1
                        #stoptime[i] = t
                        #stoptime[i] = step-1
                        #print(step-1)
                else:
                    step += 1

        return np.average(prices)

