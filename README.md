# PyTorch QA Validation Harness

This project demonstrates a QA-focused approach to testing machine learning model outputs using PyTorch.

## Objective

Instead of focusing on model building alone, this project treats ML models as systems under test and validates their outputs against expected behavior.

## Features

- Train a simple regression model (Celsius → Fahrenheit)
- Validate predictions against expected values
- Apply tolerance-based assertions
- Identify failure cases and deviations

## Setup

```bash
pip install -r requirements.txt