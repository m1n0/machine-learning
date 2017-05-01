# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualising results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.scatter(X, y, color = 'red')
# Linear regression
plt.plot(X, lin_reg.predict(X))
# Polynomial regression
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)))

# Predicting a result
y_pred_lin = lin_reg.predict(6.5)
y_pred_poly = lin_reg2.predict(poly_reg.fit_transform(6.5))