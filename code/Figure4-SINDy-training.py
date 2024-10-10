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

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
Z = Z.reshape(-1, 1)

dt = 1
dZ_dt = np.diff(Z, axis=0) / dt

X = X[:-1]
Y = Y[:-1]

# Constructing a polynomial feature library
poly = PolynomialFeatures(degree=2, include_bias=False)
XY = np.hstack([X, Y, Z[:-1]])
Theta = poly.fit_transform(XY)

# Use the first 3000 data points for model training
Theta_train = Theta[:3000, :]
dZ_dt_train = dZ_dt[:3000]

# Use the 3000th to 6000th data for testing
Theta_test = Theta[3000:6000, :]
dZ_dt_test = dZ_dt[3000:6000]

# Ridge
ridge = Ridge(alpha=0.1)
ridge.fit(Theta_train, dZ_dt_train)


coefficients = ridge.coef_[0]
intercept = ridge.intercept_
features = poly.get_feature_names_out(input_features=['X', 'Y', 'Z'])

def print_matrices(coefficients, features):
    print("Coefficient Matrix:")
    print(np.array(coefficients).reshape(-1, 1))
    print("Feature Matrix:")
    print(np.array(features).reshape(-1, 1))

print_matrices(coefficients, features)

# print
def print_polynomial_expression(coefficients, features):
    expression = " + ".join(f"{coef:.10e}*{feat}" for coef, feat in zip(coefficients, features))
    return expression

print("dZ/dt(X, Y, Z) = ", print_polynomial_expression(coefficients, features))

dZ_dt_pred = ridge.predict(Theta_test)

loss = mean_squared_error(dZ_dt_test, dZ_dt_pred)

# plot
plt.figure(figsize=(10, 6))
plt.plot(dZ_dt_test, label='Actual Power', color='b')
plt.plot(dZ_dt_pred, label='Predicted Power', linestyle='--', color='r')
plt.xlabel('Time (*300s)')
plt.ylabel('Power')
plt.title(f'dZ/dt Prediction (MSE={loss:.4f})')
plt.legend()

plt.savefig("C:\\Users\\192052\\Desktop\\SINDy-dZdt-predict.png", dpi=300)
plt.show()

