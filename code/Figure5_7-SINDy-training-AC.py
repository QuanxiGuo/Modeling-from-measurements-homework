import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

file_path = r"C:\Users\192052\Desktop\data1.csv"
data = pd.read_csv(file_path, header=0)
Wea_data = pd.read_csv(r"C:\Users\192052\Desktop\BANGKOK_Weather_2018-2019.csv")

Tem = Wea_data['T2M']
Tem = Tem.iloc[11304:11425].values

from scipy.interpolate import interp1d
hours = np.arange(len(Tem))
f = interp1d(hours, Tem, kind='linear')
minutes = np.linspace(0, len(Tem)-1, (len(Tem)-1) * 60 + 1)
Tem_interpolated = f(minutes)
Tem_interpolated = Tem_interpolated[:-1]

T = data['T'].values
H = data['H'].values
P = data['P'].values
M = Tem_interpolated

T = np.nan_to_num(T, nan=np.nanmean(T))
H = np.nan_to_num(H, nan=np.nanmean(H))
P = np.nan_to_num(P, nan=np.nanmean(P))
M = np.nan_to_num(M, nan=np.nanmean(M))

dt = 300
dT_dt = np.diff(T) / dt
dH_dt = np.diff(H) / dt
dP_dt = np.diff(P) / dt
T = T[:-1]
H = H[:-1]
P = P[:-1]
M = M[:-1]

T = T.reshape(-1, 1)
H = H.reshape(-1, 1)
P = P.reshape(-1, 1)
M = M.reshape(-1, 1)

# 构建多项式特征库，针对每个方程分别使用 T 和 M, H 和 M, P 和 M
poly_T = PolynomialFeatures(degree=2, include_bias=False)
Theta_T = poly_T.fit_transform(np.hstack([T, M]))

poly_H = PolynomialFeatures(degree=2, include_bias=False)
Theta_H = poly_H.fit_transform(np.hstack([H, M]))

poly_P = PolynomialFeatures(degree=2, include_bias=False)
Theta_P = poly_P.fit_transform(np.hstack([P, M]))

# Split training set and test set
split_idx = 3000
Theta_T_train, Theta_T_test = Theta_T[:split_idx, :], Theta_T[split_idx:, :]
Theta_H_train, Theta_H_test = Theta_H[:split_idx, :], Theta_H[split_idx:, :]
Theta_P_train, Theta_P_test = Theta_P[:split_idx, :], Theta_P[split_idx:, :]

dT_train, dT_test = dT_dt[:split_idx], dT_dt[split_idx:]
dH_train, dH_test = dH_dt[:split_idx], dH_dt[split_idx:]
dP_train, dP_test = dP_dt[:split_idx], dP_dt[split_idx:]

# Ridge
ridge_T = Ridge(alpha=0.1)
ridge_T.fit(Theta_T_train, dT_train)

ridge_H = Ridge(alpha=0.1)
ridge_H.fit(Theta_H_train, dH_train)

ridge_P = Ridge(alpha=0.1)
ridge_P.fit(Theta_P_train, dP_train)

coefficients_T = ridge_T.coef_
coefficients_H = ridge_H.coef_
coefficients_P = ridge_P.coef_

intercept_T = ridge_T.intercept_
intercept_H = ridge_H.intercept_
intercept_P = ridge_P.intercept_

features_T = poly_T.get_feature_names_out(input_features=['T', 'M'])
features_H = poly_H.get_feature_names_out(input_features=['H', 'M'])
features_P = poly_P.get_feature_names_out(input_features=['P', 'M'])

def print_polynomial_expression(coefficients, intercept, features):
    terms = " + ".join(f"{coef:.10e}*{feat}" for coef, feat in zip(coefficients, features))
    return f"{intercept:.10e} + {terms}"

print("dT/dt(T, M) = ", print_polynomial_expression(coefficients_T, intercept_T, features_T))
print("dH/dt(H, M) = ", print_polynomial_expression(coefficients_H, intercept_H, features_H))
print("dP/dt(P, M) = ", print_polynomial_expression(coefficients_P, intercept_P, features_P))

# Predicted value
dT_pred = ridge_T.predict(Theta_T_test)
dH_pred = ridge_H.predict(Theta_H_test)
dP_pred = ridge_P.predict(Theta_P_test)

loss_T = mean_squared_error(dT_test, dT_pred)
loss_H = mean_squared_error(dH_test, dH_pred)
loss_P = mean_squared_error(dP_test, dP_pred)

plt.figure(figsize=(10, 6))
plt.plot(dT_test, label='Actual Temperature', color='b')
plt.plot(dT_pred, label='Predicted Temperature', linestyle='--', color='r')
plt.xlabel('Time (*60s)')
plt.ylabel('Temperature')
plt.title(f'Temperature Prediction')
plt.legend()
plt.savefig("C:\\Users\\192052\\Desktop\\SINDy-dTdt.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(dH_test, label='Actual Humidity', color='b')
plt.plot(dH_pred, label='Predicted Humidity', linestyle='--', color='r')
plt.xlabel('Time (*60s)')
plt.ylabel('Humidity')
plt.title(f'Humidity Prediction')
plt.legend()
plt.savefig("C:\\Users\\192052\\Desktop\\SINDy-dHdt.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(dP_test, label='Actual Power', color='b')
plt.plot(dP_pred, label='Predicted Power', linestyle='--', color='r')
plt.xlabel('Time (*60s)')
plt.ylabel('Power')
plt.title(f'Power Prediction')
plt.legend()
plt.savefig("C:\\Users\\192052\\Desktop\\SINDy-dPdt.png", dpi=300)
plt.show()
