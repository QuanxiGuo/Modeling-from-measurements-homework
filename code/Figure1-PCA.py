import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = r"C:\Users\192052\Desktop\pv_solar.csv"
data = pd.read_csv(file_path)

# Perform PCA to retain three-dimensional features
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data)
pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2', 'PCA3'])

# Randomly sample the data (show only 3% of the data points)
sampled_pca_df = pca_df.sample(frac=0.03,random_state=42)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(sampled_pca_df['PCA1'], sampled_pca_df['PCA2'], sampled_pca_df['PCA3'], c='blue', marker='o')
ax.set_xlabel('PCA1', color='r')
ax.set_ylabel('PCA2', color='r')
ax.set_zlabel('PCA3', color='r')

plt.savefig('PCA.png',dpi=300)
plt.show()
