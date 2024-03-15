from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import LinearRegression

df_input = pd.read_csv("returns.csv")
df_output = pd.read_csv("index_returns.csv")