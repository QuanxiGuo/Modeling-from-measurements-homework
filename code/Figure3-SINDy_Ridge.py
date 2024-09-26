import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sympy import symbols
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

file_path = r"C:\\Users\\192052\\Desktop\\pv_solar1.xlsx"
data = pd.read_excel(file_path)

X = pd.to_numeric(data['Temperature'], errors='coerce')
Y = pd.to_numeric(data['Radiation'], errors='coerce')
Z = pd.to_numeric(data['Power'], errors='coerce')

# Transpose the matrix
Theta = np.vstack([X, Y]).T
gamma = Z.values

# Building a library
poly = PolynomialFeatures(degree=3, include_bias=False)
Theta = poly.fit_transform(Theta)

# Using Lasso sparse regression model
ridge = Ridge(alpha=0.1)
ridge.fit(Theta, gamma)

coefficients = ridge.coef_
intercept = ridge.intercept_
features = poly.get_feature_names_out(['X', 'Y'])

# Function to print coefficients and features as matrices
def print_matrices(coefficients, features):
    print("Coefficient Matrix:")
    print(np.array(coefficients).reshape(-1, 1))
    print("Feature Matrix:")
    print(np.array(features).reshape(-1, 1))

print_matrices(coefficients, features)


# Constructing polynomial expressions
def print_polynomial_expression(coefficients, features):
    expression = " + ".join(f"{coef:.10e}*{feat}" for coef, feat in zip(coefficients, features))
    return expression

print("Z(X, Y) = ", print_polynomial_expression(coefficients, features))

# predict power
power_pred = ridge.predict(Theta)
loss = mean_squared_error(gamma, power_pred)

#plot
plt.figure(figsize=(10, 6))

plt.plot(gamma, label='Power Actual', color='b')
plt.plot(power_pred, '--', label='Power Predicted', color='r')
plt.xlabel('Time (*300s)')
plt.ylabel('Power')
plt.title(f'Power Prediction with Ridge Regression (MSE={loss:.4f})')
plt.legend()

plt.tight_layout()
plt.savefig("C:\\Users\\192052\\Desktop\\SINDy-Ridge.png", dpi=300)
plt.show()
