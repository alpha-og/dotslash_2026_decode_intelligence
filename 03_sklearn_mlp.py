import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_true = np.array([[0], [1], [1], [0]], dtype=float).ravel()

model = MLPClassifier(hidden_layer_sizes=(4,), activation="relu",
                      max_iter=3000, random_state=1)
model.fit(X, y_true)   # this is the whole training loop from script 2, done by the library

y_pred = model.predict(X)
print(f"predictions after training: {y_pred} (true answer {y_true})")
print(f"accuracy: {model.score(X, y_true):.1%}")