import numpy as np
import matplotlib.pyplot as plt
import sklearn
def derivative(x):
    return x*(1.0-x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
x_result = []
y_result = []


with open('data.csv') as f:
    for line in f:
        curr = line.split(',')
        new_curr = [1]
        for item in curr[:len(curr) - 1]:
            new_curr.append(float(item))
        x_result.append(new_curr)
        y_result.append([float(curr[-1])])
x_result = np.array(x_result)
x_result = preprocessing.scale(x_result)
y_result = np.array(y_result)
