import pandas as pd
import gradient_descent as gd
import numpy as np

df = pd.read_csv("Machine Learning/Gradient Descent/test_scores.csv")

x = np.array(df["math"])
y = np.array(df["cs"])

t = gd.descend(x, y)
