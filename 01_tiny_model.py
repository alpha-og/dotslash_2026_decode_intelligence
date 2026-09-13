def predict(x, w, b):
    return w * x + b  # linear model: weight x input + bias

w, b = 0.0, 0.0

x = 5
y_true = 50

y_pred = predict(x, w, b)
print("first prediction:", y_pred)

loss = (y_true - y_pred) ** 2  # squared error: how wrong the guess is
print("loss:", loss)

learning_rate = 0.01

for step in range(50):
    y_pred = predict(x, w, b)
    loss = (y_true - y_pred) ** 2  # squared error

    grad_w = -2 * x * (y_true - y_pred)  # derivative of loss with respect to w
    grad_b = -2 * (y_true - y_pred)      # derivative of loss with respect to b

    w -= learning_rate * grad_w  # nudge w against the gradient
    b -= learning_rate * grad_b  # nudge b against the gradient

    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}, w={w:7.3f}, b={b:7.3f}, loss={loss:8.2f}")

print(f"\nprediction after training: {predict(x, w, b):.2f} (true answer {y_true})")