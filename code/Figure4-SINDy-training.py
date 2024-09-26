import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# read data
file_path = "C:\\Users\\192052\\Desktop\\pv_solar1.xlsx"
data = pd.read_excel(file_path, header=0)

X = data['Temperature'].values
Y = data['Radiation'].values
Z = data['Power'].values

# Convert the data into one column
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
Z = Z.reshape(-1, 1)

# Building a library
poly = PolynomialFeatures(degree=3, include_bias=False)
XY = np.hstack([X, Y])
Theta = poly.fit_transform(XY)

# Use the first 3000 data points to train the model
Theta1 = Theta [: 3000,:]
Z1 = Z [: 3000, :]

# Use the 3000th to 6000th data to test the accuracy of the trained model
Theta2 = Theta [3000: 6000, :]
Z2 = Z [3000 : 6000, :]

# Use Ridge regression to fit Z as a polynomial function of X and Y
ridge = Ridge(alpha=0.1)
ridge.fit(Theta1, Z1)

coefficients = ridge.coef_[0]
intercept = ridge.intercept_[0]
features = poly.get_feature_names_out(input_features=['X', 'Y'])


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

# predicted value
Z_p = ridge.predict(Theta2)
loss = mean_squared_error(Z2, Z_p)

# plot
plt.figure(figsize=(10, 6))
plt.plot(Z2, label='Actual power', color='b')
plt.plot(Z_p, label='Predicted power', linestyle='--', color='r')
plt.xlabel('Time (*300s)')
plt.ylabel('Power')
plt.title(f'Power Prediction (MSE={loss:.4f})')
plt.legend()

plt.savefig("C:\\Users\\192052\\Desktop\\SINDy-predict.png", dpi=300)
plt.show()
