# MNIST Neural Network From Scratch

A fully connected neural network built **entirely from scratch** using **NumPy** for handwritten digit classification on the MNIST dataset.

The goal of this project is to implement every component of a modern neural network manually—including forward propagation, backpropagation, loss computation, optimization, and weight updates—without relying on deep learning frameworks such as TensorFlow, PyTorch, Keras, or Scikit-Learn.

This repository documents the complete development process, from data preprocessing to model training, evaluation, visualization, and prediction generation.
---

## Features

- Neural Network implemented completely from scratch
- Two Hidden Layers (512 neurons each)
- ReLU Activation Function
- Softmax Output Layer
- Categorical Cross Entropy Loss
- Adam Optimizer (implemented from scratch)
- He Weight Initialization
- Mini-Batch Gradient Descent
- Learning Rate Scheduler
- Early Stopping
- L2 Regularization
- Gradient Clipping
- Validation Accuracy Tracking
- Matplotlib Training Visualizations
- Automatic Kaggle Submission File Generation

---

## Network Architecture

```
Input Layer (784)

        │

        ▼

Hidden Layer 1 (512)
ReLU

        │

        ▼

Hidden Layer 2 (512)
ReLU

        │

        ▼

Output Layer (10)
Softmax
```

---

## Dataset

The project uses the **MNIST Handwritten Digits Dataset**.

Each sample consists of:

- 28 × 28 grayscale image
- Flattened into **784 input features**
- Pixel values ranging from **0–255**
- Labels from **0–9**

The dataset is provided in CSV format:

```
competition_train.csv
competition_test.csv
sample_submission.csv
```

---

## Project Structure

```
mnist-neural-network-from-scratch/
│
├── data/
│   ├── competition_train.csv
│   ├── competition_test.csv
│   └── sample_submission.csv
│
├── plots/
│
├── submissions/
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Training Pipeline

```
Load Dataset
      │
Normalize Pixel Values
      │
One-Hot Encode Labels
      │
Train / Validation Split
      │
Initialize Weights (He Initialization)
      │
Mini-Batch Training
      │
Forward Propagation
      │
Cross Entropy Loss
      │
Backpropagation
      │
Adam Optimization
      │
Validation
      │
Learning Rate Scheduling
      │
Save Best Model
```

---

## Optimizations

To maximize classification accuracy, the following optimization techniques are implemented:

- He Initialization
- Adam Optimizer
- Mini-Batch Gradient Descent
- Learning Rate Scheduling
- Early Stopping
- L2 Regularization
- Gradient Clipping
- Stable Softmax Implementation
- Numerically Stable Cross Entropy Loss
- Data Shuffling Every Epoch

---

## Training Visualizations

During training, Matplotlib is used to visualize:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

These plots help monitor convergence and detect overfitting.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib

---

## Goal

The objective of this project is to build a highly optimized neural network capable of achieving **99%+ classification accuracy** on the MNIST handwritten digit dataset while implementing every component of the learning algorithm manually.

---

## Future Improvements

Potential future enhancements include:

- Dropout
- Batch Normalization
- Xavier Initialization
- Learning Rate Warmup
- AdamW Optimizer
- Convolutional Neural Networks (CNNs)
- GPU Acceleration

---