import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron_scratch import Perceptron

# Learning model on Iris Dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

print(X)
print(y)

ppn_clf = Perceptron(learning_rate=0.1, epochs=10)
errors = ppn_clf.fit(X, y)

# Testing
print(ppn_clf.predict(np.array([5.1, 1.4])))

plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
