# Beginner Documentation for `train_model.py`

This document explains the PyTorch training script line by line for someone new to `torch`, `torch.nn`, and machine learning.

## What the script does

The script trains a tiny PyTorch model to learn Celsius to Fahrenheit conversion.

The real-world formula is:

```text
Fahrenheit = Celsius * 1.8 + 32
```

But instead of directly coding that formula, the model learns the relationship from examples.

---

## Original script

```python
import torch
import torch.nn as nn
from pathlib import Path

MODEL_PATH = Path("../temperature_model.pth")

celsius = torch.tensor([[-40.0], [0.0], [10.0], [20.0], [30.0], [40.0]])
fahrenheit = torch.tensor([[-40.0], [32.0], [50.0], [68.0], [86.0], [104.0]])

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
```

---

## Line-by-line explanation

### `import torch`

Imports the main PyTorch library.

You use `torch` for:
- tensors
- math operations
- random numbers
- saving and loading model weights
- GPU/CPU handling

A tensor is the basic data structure in PyTorch.

---

### `import torch.nn as nn`

Imports PyTorch's neural network module.

`torch.nn` contains building blocks for models, such as:
- `nn.Linear`
- `nn.MSELoss`
- `nn.ReLU`
- `nn.Sequential`

The alias `nn` is standard PyTorch style.

---

### `from pathlib import Path`

Imports Python's `Path` class.

`Path` helps create file paths in a cleaner way than plain strings.

---

### `MODEL_PATH = Path("../temperature_model.pth")`

Defines where the trained model weights will be saved.

`..` means one folder up.

`.pth` is a common extension for saved PyTorch weights.

---

### `celsius = torch.tensor([...])`

Creates the input training data.

```python
import torch
celsius = torch.tensor([[-40.0], [0.0], [10.0], [20.0], [30.0], [40.0]])
```

This is a tensor with shape:

```text
[6, 1]
```

That means:
- 6 examples
- 1 feature per example

The feature is Celsius temperature.

---

### `fahrenheit = torch.tensor([...])`

Creates the expected output values.

```python
import torch
fahrenheit = torch.tensor([[-40.0], [32.0], [50.0], [68.0], [86.0], [104.0]])
```

This tensor also has shape:

```text
[6, 1]
```

Each Fahrenheit value corresponds to the Celsius value at the same row.

---

### `fahrenheit = fahrenheit + torch.randn_like(fahrenheit) * 0.5`

Adds small random noise to the expected Fahrenheit values.

`torch.randn_like(fahrenheit)` creates random values with the same shape as `fahrenheit`.

Multiplying by `0.5` keeps the noise small.

This simulates imperfect real-world data.

Example:

```text
32.0 may become 31.8 or 32.3
```

---

### `model = nn.Linear(1, 1)`

Creates a linear model.

This means:

```text
1 input value -> 1 output value
```

The model internally uses:

```text
output = input * weight + bias
```

For Celsius to Fahrenheit, the ideal values are:

```text
weight ≈ 1.8
bias ≈ 32
```

At first, PyTorch initializes them randomly. Training changes them.

---

### `loss_fn = nn.MSELoss()`

Creates the loss function.

`MSELoss` means Mean Squared Error Loss.

It measures how far predictions are from expected values.

It is commonly used for regression, where the model predicts numbers.

Lower loss means better predictions.

---

### `optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)`

Creates the optimizer.

The optimizer updates the model's learnable values.

For this model, the learnable values are:
- weight
- bias

`model.parameters()` gives those values to the optimizer.

`SGD` means Stochastic Gradient Descent.

`lr=0.0001` is the learning rate.

The learning rate controls how large each update step is.

---

### `for epoch in range(50000):`

Starts the training loop.

An epoch is one training pass.

This loop runs 50,000 update cycles.

---

### `prediction = model(celsius)`

Runs the model on the Celsius inputs.

This is called the forward pass.

The model produces predicted Fahrenheit values.

---

### `loss = loss_fn(prediction, fahrenheit)`

Compares the predictions to the expected Fahrenheit values.

The result is a single loss value.

The training goal is to reduce this number.

---

### `optimizer.zero_grad()`

Clears old gradients.

PyTorch accumulates gradients by default.

If you do not clear them, new gradients are added on top of old gradients.

This line is required in most PyTorch training loops.

---

### `loss.backward()`

Runs backpropagation.

This calculates gradients.

A gradient tells PyTorch:

```text
How should each parameter change to reduce the loss?
```

For this model, gradients are calculated for:
- weight
- bias

---

### `optimizer.step()`

Updates the model parameters.

This is where learning actually happens.

The optimizer uses the gradients to slightly change the weight and bias.

Over many epochs, the model gets better.

---

### `torch.save(model.state_dict(), MODEL_PATH)`

Saves the learned model weights.

`model.state_dict()` contains the trained parameters.

For this model, it contains:
- weight
- bias

This saves only the model's learned values, not the entire Python model object.

---

### `print("Model trained and saved.")`

Prints a confirmation message.

---

### `print("Weight:", model.weight.item())`

Prints the learned weight.

`.item()` converts a one-value tensor into a normal Python number.

A successful model should learn a weight close to:

```text
1.8
```

---

### `print("Bias:", model.bias.item())`

Prints the learned bias.

A successful model should learn a bias close to:

```text
32
```

---

## Important PyTorch concepts used

### Tensor

A tensor is a container for numbers.

Examples:

torch.tensor([1.0, 2.0, 3.0])

Tensors can be:
- scalar: one number
- vector: list of numbers
- matrix: table of numbers
- higher-dimensional data

---

### Shape

Shape tells you the structure of the tensor.

Example:

torch.tensor([[1.0], [2.0], [3.0]])

Shape:

```text
[3, 1]
```

Meaning:
- 3 rows
- 1 column

In ML terms:
- 3 samples
- 1 feature

---

### `nn.Linear`

A linear layer learns:

```text
output = input * weight + bias
```

In this script:

nn.Linear(1, 1)


means:
- 1 input feature
- 1 output value

---

### `MSELoss`

Mean Squared Error Loss measures numeric prediction error.

It is good for regression.

The model is penalized more for larger mistakes.

---

### Optimizer

An optimizer updates model parameters.

In this script:

torch.optim.SGD(...)

uses gradient descent to improve the model.

---

### Backpropagation

Backpropagation calculates how much each parameter contributed to the error.

In PyTorch, this happens with:

loss.backward()


---

### Gradient

A gradient tells the optimizer:
- which direction to move a parameter
- how strongly to move it

---

### Learning rate

The learning rate controls update size.

A high learning rate can overshoot.

A low learning rate can train slowly.

---

### `state_dict`

A `state_dict` is a dictionary of learned parameters.

For this model, it stores:
- weight
- bias

---

## Training loop summary

Every epoch does this:

```text
1. Predict
2. Measure error
3. Clear old gradients
4. Calculate new gradients
5. Update weight and bias
```

In code:


prediction = model(celsius)
loss = loss_fn(prediction, fahrenheit)

optimizer.zero_grad()
loss.backward()
optimizer.step()


---

## QA interpretation

From a QA point of view, this script is testing a learning process.

You can inspect:
- Did loss decrease?
- Did the learned weight approach 1.8?
- Did the learned bias approach 32?
- Do predictions stay within tolerance?
- Does retraining create drift?

That is the bridge between PyTorch and quality engineering.
