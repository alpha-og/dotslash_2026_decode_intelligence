import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # squashes to 0-1, like a probability


def relu(z):
    return np.maximum(0, z)  # the bend: zero out negatives


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_true = np.array([[0], [1], [1], [0]], dtype=float)

learning_rate = 1.0

np.random.seed(1)
W1 = np.random.randn(2, 4) * 0.5  # 2 inputs -> 4 hidden neurons
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5  # 4 hidden neurons -> 1 output
b2 = np.zeros((1, 1))

h = relu(X @ W1 + b1)  # hidden layer: lines turned into switches
y_pred = sigmoid(h @ W2 + b2)  # combine switches, squash to 0-1
print("first prediction (with bend):", np.round(y_pred.ravel(), 2))

loss = np.mean((y_true - y_pred) ** 2)
print("loss:", loss)

for step in range(3000):
    h = relu(X @ W1 + b1)
    y_pred = sigmoid(h @ W2 + b2)
    loss = np.mean((y_true - y_pred) ** 2)

    d = (y_pred - y_true) * y_pred * (1 - y_pred)  # error through sigmoid's derivative
    dH = (d @ W2.T) * (h > 0)  # error back through ReLU's derivative

    grad_W2 = h.T @ d  # derivative of loss with respect to W2
    grad_b2 = d.sum(0, keepdims=True)  # derivative of loss with respect to b2
    grad_W1 = X.T @ dH  # derivative of loss with respect to W1
    grad_b1 = dH.sum(0, keepdims=True)  # derivative of loss with respect to b1

    W2 -= learning_rate * grad_W2  # nudge W2 against the gradient
    b2 -= learning_rate * grad_b2  # nudge b2 against the gradient
    W1 -= learning_rate * grad_W1  # nudge W1 against the gradient
    b1 -= learning_rate * grad_b1  # nudge b1 against the gradient

    if step % 500 == 0 or step == 2999:
        print(f"step {step:4d}, loss={loss:.4f}")

final_pred = sigmoid(relu(X @ W1 + b1) @ W2 + b2)
print(
    f"\nprediction after training (with bend): {np.round(final_pred.ravel(), 2)} (true answer {y_true.ravel()})"
)
