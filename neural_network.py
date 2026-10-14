import numpy as np
import matplotlib.pyplot as plt
import torch

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

print("Softmax ([2.0, 1.0, 0.1]) =", np.round(softmax(np.array([2.0, 1.0, 0.1])), 3))

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([[0], [1], [1], [0]], dtype=float)


np.random.seed(7)

W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

learning_rate = 0.5
epochs = 5000
loss_history = []

m = len(X)


for epoch in range(epochs):

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    loss = cross_entropy(y, A2)
    loss_history.append(loss)

    dZ2 = (A2 - y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

Z1 = X @ W1 + b1
A1 = relu(Z1)
A2 = sigmoid(A1 @ W2 + b2)

print("\nXOR predictions:")
print(np.round(A2.flatten(), 3))

print("\nPredicted classes:")
print((A2.flatten() >= 0.5).astype(int))


plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("XOR Neural Network Training Loss")
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()


def train_with_rate(lr, epochs=5000):
    np.random.seed(42)

    w1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros((1, 4))
    w2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))

    losses = []

    for epoch in range(epochs):
        z1 = X @ w1 + b1
        a1 = relu(z1)
        a2 = sigmoid(a1 @ w2 + b2)

        loss = cross_entropy(y, a2)
        losses.append(loss)

        dz2 = (a2 - y) / len(X)
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ w2.T) * (z1 > 0)
        dw1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        w2 -= lr * dw2
        b2 -= lr * db2
        w1 -= lr * dw1
        b1 -= lr * db1

        if not np.isfinite(loss):
            break
    return losses


plt.figure()

for lr in [0.01, 0.5, 5.0]:
    losses = train_with_rate(lr)
    plt.plot(losses, label=f"Learning rate = {lr}")

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Effect of Learning Rate")
plt.legend()
plt.grid(True)
plt.savefig("learning_rate_comparison.png")
plt.show()


w = 1
x = 2
b = 0
t = 10

prediction = w * x + b
error = prediction - t

dw = 2 * error * x
db = 2 * error

print("\nManual gradients:")
print("dw =", dw)
print("db =", db)


w_t = torch.tensor(1.0, requires_grad=True)
b_t = torch.tensor(0.0, requires_grad=True)

x_t = torch.tensor(2.0)
t_t = torch.tensor(10.0)

loss_t = (w_t * x_t + b_t - t_t) ** 2
loss_t.backward()

print("\nPyTorch autograd:")
print("w.grad =", w_t.grad.item())
print("b.grad =", b_t.grad.item())


