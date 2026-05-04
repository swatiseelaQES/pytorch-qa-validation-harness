import torch
import torch.nn as nn
from pathlib import Path

MODEL_PATH = Path("temperature_model.pth")


def load_model():
    model = nn.Linear(1, 1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def predict_fahrenheit(model, celsius_value: float) -> float:
    input_tensor = torch.tensor([[celsius_value]], dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    return prediction