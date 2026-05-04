import torch
import torch.nn as nn
from pathlib import Path

MODEL_PATH = Path("../temperature_model.pth")

celsius = torch.tensor([[-40.0], [0.0], [10.0], [20.0], [30.0], [40.0]])
fahrenheit = torch.tensor([[-40.0], [32.0], [50.0], [68.0], [86.0], [104.0]])

# Add slight noise to simulate real-world imperfect data
fahrenheit = fahrenheit + torch.randn_like(fahrenheit) * 0.5

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(50000):
    prediction = model(celsius)
    loss = loss_fn(prediction, fahrenheit)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), MODEL_PATH)

print("Model trained and saved.")
print("Weight:", model.weight.item())
print("Bias:", model.bias.item())