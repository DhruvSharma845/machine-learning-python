from adaptive_linear_neuron_scratch import Adaline
import numpy as np
import matplotlib.pyplot as plt
from iris_prep import X_std, y

# Learning model on Iris Dataset

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada_clf_1 = Adaline(epochs=10, learning_rate=0.01)
ada_clf_1.fit(X_std, y)
ax[0].plot(range(1, len(ada_clf_1.cost) + 1), np.log10(ada_clf_1.cost), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada_clf_2 = Adaline(epochs=10, learning_rate=0.0001)
ada_clf_2.fit(X_std, y)
ax[1].plot(range(1, len(ada_clf_2.cost) + 1), ada_clf_2.cost, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
