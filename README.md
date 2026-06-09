# Deep Learning — CNN, BERT & Neuromorphic Vision

Hands-on notebooks exploring Convolutional Neural Networks (CNNs), transformer-based encoders (BERT), and neuromorphic image classification using event-camera data with the N-MNIST dataset.

---

## Repository Structure

```
Deep_Learning_CNN/
├── Cat_Dog_CNN.ipynb                  ← Custom CNN for binary image classification
├── Cats_Vs_Dogs_Classification.ipynb  ← Classification CNN
├── Encoder_BERT.ipynb                 ← BERT encoder for text representation
└── N_MNIST_CNN.ipynb                  ← CNN on neuromorphic N-MNIST event data
```

---

## Notebooks

### 1. `Cat_Dog_CNN.ipynb`
A custom CNN trained from scratch to classify images as cats or dogs.

- **Architecture:** Convolutional layers → MaxPooling → Fully Connected → Softmax
- **Dataset:** Binary image dataset (cats and dogs)
- **Key concepts:** Feature extraction, activation functions, overfitting control via dropout

---

### 2. `Cats_Vs_Dogs_Classification.ipynb`
CNN for Cats vs Dogs classification problem.

- **Approach:** CNN 
- **Dataset:** Cats vs Dogs (Kaggle)

---

### 3. `Encoder_BERT.ipynb`
Exploration of the BERT encoder for generating contextual text embeddings.

- **Model:** `bert-base-uncased` via HuggingFace Transformers
- **Key concepts:** Tokenization, attention masks, `[CLS]` token embeddings, sentence similarity
- **Use cases:** Semantic search, text classification, feature extraction for downstream tasks

---

### 4. `N_MNIST_CNN.ipynb`
CNN trained on the N-MNIST dataset — a neuromorphic version of MNIST captured using an event camera (Dynamic Vision Sensor).

- **Dataset:** N-MNIST (event-based, 70,000 samples, 10 digit classes)
- **Approach:** Converting event streams into frame-based representations for CNN input
- **Key concepts:** Neuromorphic data preprocessing, event-to-frame conversion, standard CNN pipeline on non-standard data

---

## Tech Stack

PyTorch / TensorFlow · HuggingFace Transformers · NumPy · Matplotlib · Python

---

## Setup

```bash
pip install torch torchvision transformers numpy matplotlib
```
