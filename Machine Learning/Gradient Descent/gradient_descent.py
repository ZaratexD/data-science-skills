"""
Finds optimized coefficients m and b in y = mx + b for line of best fit.
Does so by finding minimum using gradient descent.
Essential what sklearn.linear_model does 
"""

import numpy as np

initial_m = 1
iterations = 15
step = 0.01


def gradient_descent(x, y):
    m_curr = b_curr = 0
    m_curr = initial_m
    residuals = np.zeros(16)

    for i in range(iterations):
        y_predict = x * m_curr + b_curr

        sum_of_squares = y - y_predict
        z = sum([val**2 for val in sum_of_squares])
        residuals[i] = z

        tang = (residuals[i + 1] - residuals[i]) / step
        m_curr = m_curr - tang
        print(m_curr)

    return (m_curr, b_curr)


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)
