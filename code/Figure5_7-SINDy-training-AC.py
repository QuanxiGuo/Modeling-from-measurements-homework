import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import interp1d
from sympy import symbols
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

file_path = r"C:\Users\192052\Desktop\data1.csv"
data = pd.read_csv(file_path, header=0)

Wea_data= pd.read_csv(r"C:\Users\192052\Desktop\BANGKOK_Weather_2018-2019.csv")
Tem= Wea_data['T2M']

Tem = Tem.iloc[11304:11425].values

# Get temperature data every minute by interpolation
hours = np.arange(len(Tem))
f = interp1d(hours, Tem, kind='linear')
minutes = np.linspace(0, len(Tem)-1, (len(Tem)-1) * 60 + 1)
Tem_interpolated = f(minutes)
Tem_interpolated = Tem_interpolated [: -1]

T = data['T'].values
H = data['H'].values
P = data['P'].values
M = Tem_interpolated

T = np.nan_to_num(T, nan=np.nanmean(T))
H = np.nan_to_num(H, nan=np.nanmean(H))
P = np.nan_to_num(P, nan=np.nanmean(P))
M = np.nan_to_num(M, nan=np.nanmean(M))

# Convert the data into one column
T = T.reshape(-1, 1)
H = H.reshape(-1, 1)
P = P.reshape(-1, 1)
M = M.reshape(-1, 1)

# Building a library
poly = PolynomialFeatures(degree=3, include_bias=False)
# TH = np.hstack([M, H])
Theta = poly.fit_transform(M)
Theta1 = Theta [: 3000,:]
Theta2 = Theta [3000: 6000, :]

P = np.hstack([T, P, H])
P1 = P [: 3000, :]
P2 = P [3000 : 6000, :]



# Use Ridge regression to fit Z as a polynomial function of X and Y
ridge = Ridge(alpha=0.1)
ridge.fit(Theta1, P1)

coefficients = ridge.coef_[0]
intercept = ridge.intercept_[0]
features = poly.get_feature_names_out(input_features=['M'])


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

print("P(T, H) = ", print_polynomial_expression(coefficients, features))

# predicted value
P_p = ridge.predict(Theta2)
loss1 = mean_squared_error(P2[:,0], P_p[:,0])
loss2 = mean_squared_error(P2[:,1], P_p[:,1])
loss3 = mean_squared_error(P2[:,2], P_p[:,2])

# plot
plt.figure(figsize=(10, 6))
plt.plot(P2[:,0], label='Actual power', color='b')
plt.plot(P_p[:,0], label='Predicted power', linestyle='--', color='r')
plt.xlabel('Time (*60s)')
plt.ylabel('Power (kW)')
plt.title(f'Power Prediction (MSE={loss1:.4f})')
plt.savefig("C:\\Users\\192052\\Desktop\\SINDy-power.png", dpi=300)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(P2[:,1], label='Actual temperature', color='b')
plt.plot(P_p[:,1], label='Predicted temperature', linestyle='--', color='r')
plt.xlabel('Time (*60s)')
plt.ylabel('Temperature')
plt.title(f'Temperature Prediction (MSE={loss2:.4f})')
plt.savefig("C:\\Users\\192052\\Desktop\\SINDy-Tem.png", dpi=300)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(P2[:,2], label='Actual humidity', color='b')
plt.plot(P_p[:,2], label='Predicted humidity', linestyle='--', color='r')
plt.xlabel('Time (*60s)')
plt.ylabel('Humidity')
plt.title(f'Humidity Prediction (MSE={loss3:.4f})')
plt.savefig("C:\\Users\\192052\\Desktop\\SINDy-Hum.png", dpi=300)
plt.legend()
plt.show()
