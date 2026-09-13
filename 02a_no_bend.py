import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # squashes to 0-1, like a probability


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_true = np.array([[0], [1], [1], [0]], dtype=float)

learning_rate = 1.0

np.random.seed(1)
W = np.random.randn(2, 1) * 0.5  # one neuron = one straight line
b = np.zeros((1, 1))

y_pred = sigmoid(X @ W + b)
print("first prediction (no bend):", np.round(y_pred.ravel(), 2))

loss = np.mean((y_true - y_pred) ** 2)  # mean squared error
print("loss:", loss)

for step in range(3000):
    y_pred = sigmoid(X @ W + b)
    loss = np.mean((y_true - y_pred) ** 2)

    d = (y_pred - y_true) * y_pred * (1 - y_pred)  # error through sigmoid's derivative
    grad_W = X.T @ d  # derivative of loss with respect to W
    grad_b = d.sum(0, keepdims=True)  # derivative of loss with respect to b

    W -= learning_rate * grad_W  # nudge W against the gradient
    b -= learning_rate * grad_b  # nudge b against the gradient

    if step % 500 == 0 or step == 2999:
        print(f"step {step:4d}, loss={loss:.4f}")

print(
    f"\nprediction after training (no bend): {np.round(sigmoid(X @ W + b).ravel(), 2)} (true answer {y_true.ravel()})"
)
