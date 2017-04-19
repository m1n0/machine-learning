# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plit
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
