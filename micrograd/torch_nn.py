import torch
from torch import nn

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]

# The desired targets (labels)
ys = [1.0, -1.0, -1.0, 1.0]

xst = torch.tensor(xs, dtype=torch.float32)
yst = torch.tensor(ys, dtype=torch.float32).view(-1, 1)


# 2. The Model (Equivalent to our MLP(3, [4, 4, 1]))
model = nn.Sequential(
    nn.Linear(3, 4),  # Layer 1: 3 in -> 4 out
    nn.Tanh(),  # Activation 1
    nn.Linear(4, 4),  # Layer 2: 4 in -> 4 out
    nn.Tanh(),  # Activation 2
    nn.Linear(4, 1),  # Layer 3: 4 in -> 1 out
    nn.Tanh(),  # Activation 3
)

# 3. The Loss and Optimizer
loss_fn = nn.MSELoss(reduction="sum")  # Sum of squared errors (exact same as micrograd)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# 4. The Training Loop
epochs = 160
for i in range(epochs):
    # Step 1: Forward pass (matrix multiplication happens in parallel on all 4 rows!)
    ypred = model(xst)
    # Step 2: Compute Loss
    loss = loss_fn(ypred, yst)
    # Step 3: Zero gradients & Backward pass
    optimizer.zero_grad()  # Equivalent to mlp.zero_grad()
    loss.backward()  # Autograd runs backprop!
    # Step 4: Update weights
    optimizer.step()  # Equivalent to: for p in params: p.data -= lr * p.grad
    if i % 20 == 0 or i == epochs - 1:
        print(f"step {i:3d} | loss = {loss.item():.4f}")


print("\nTarget values:     ", ys)
print("PyTorch predictions:", [round(p, 3) for p in ypred.view(-1).tolist()])
