import numpy as np
import matplotlib.pyplot as plt
from perceptron_scratch import Perceptron
from iris_prep import X, y

# Learning model on Iris Dataset

ppn_clf = Perceptron(learning_rate=0.1, epochs=10)
errors = ppn_clf.fit(X, y)

# Testing
print(ppn_clf.predict(np.array([5.1, 1.4])))

plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
