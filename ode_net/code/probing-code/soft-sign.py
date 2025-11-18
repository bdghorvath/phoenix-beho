import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def soft_sign_mod(x):
    shift = 0.5
    shifted_input = x - shift
    return shifted_input / (1 + torch.abs(shifted_input))

x = torch.linspace(-5, 5, 1000)

plt.figure(figsize=(10,6))
plt.plot(x, soft_sign_mod(x), label="soft_sign_mod", linewidth=2)
plt.plot(x, F.softsign(x), label="torch.nn.functional.softsign", linestyle="--")
plt.plot(x, torch.tanh(x), label="tanh", linestyle=":")
plt.plot(x, F.relu(x), label="ReLU", linestyle="-.")
plt.legend()
plt.title("Comparison of activation functions")
plt.grid(True)
plt.show()
