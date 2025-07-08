import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Loading datset
Data_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("IRIS.csv", skiprows=1, names=Data_names)

#Loading Samples
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Plotting the Sample for Sepal Length and Sepal Width
species = data['type'].unique()
colors = ['red', 'green', 'blue']

#intinializing the plot
plt.figure(figsize=(8, 6))
for sp, color in zip(species, colors):
    subset = data[data['type'] == sp]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], label=sp, color=color, edgecolor='k', s=60)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset: Sepal Length vs Sepal Width')
plt.legend(title='Species')
plt.grid(True)
plt.tight_layout()
plt.show()



