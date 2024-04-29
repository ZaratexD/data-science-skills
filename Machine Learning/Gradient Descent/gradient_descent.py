"""
Finds optimized coefficients m and b in y = mx + b for line of best fit.
Does so by finding minimum using gradient descent.
Essential what sklearn.linear_model does 

1. Take derivative of Loss Function for each paramater in it (Gradient of the Loss Funtion)
2. Pick random values for the parameter 
3. Plug the parameter values into the derivatives
4. Calculate Step size. Step Size = Slope * Learning ratge
5 Calculate new params New Parameters = Old Paramater - stepsize
Repeats 3-5 until close to 0 or maximum # of steps
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### TODO: use plt to see change in LOB. implement way that rate is changed automatically


def descend(x, y):
    curr_m = curr_b = 0
    rate = 0.0001
    n = x.shape[0]
    iterations = 500
    cost_g = m = b = [0] * iterations

    y_hat = curr_m * x + curr_b
    i_cost = (1 / n) * sum([i**2 for i in (y - y_hat)])
    slope_m = (-2 / n) * sum(x * (y - y_hat))
    slope_b = (-2 / n) * sum(y - y_hat)

    i = 0
    cost = i_cost + 2
    print(i_cost, cost)
    while math.isclose(i_cost, cost, rel_tol=1e-20) != True:
        if i == iterations:
            break
        i_cost = cost
        y_hat = curr_m * x + curr_b
        cost = (1 / n) * sum([i**2 for i in (y - y_hat)])
        slope_m = (-2 / n) * sum(x * (y - y_hat))
        slope_b = (-2 / n) * sum(y - y_hat)

        cost_g[i] = cost
        m[i] = curr_m
        b[i] = curr_b
        i += 1
        # way for loop to reduce redundant code
        curr_m = curr_m - slope_m * rate
        curr_b = curr_b - slope_b * rate


df = pd.read_csv("Machine Learning/Gradient Descent/test_scores.csv")

x = np.array(df["math"])
y = np.array(df["cs"])

descend(x, y)
