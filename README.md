# Classification-of-fashion-mnist-clothes-categories
🧠 Biologically Inspired AI: Fashion-MNIST Classification

This project explores biologically inspired approaches to image classification using the Fashion-MNIST dataset. A convolutional neural network (CNN) is developed and enhanced with data augmentation strategies such as Mixup and CutMix, along with class-weighted sampling to address class imbalance.

# 🚀 Features

🧹 CNN architecture tailored for 28×28 grayscale fashion images

♻️ Mixup and CutMix augmentation for regularization and better generalization

⚖️ Weighted sampling for imbalanced class distribution

🧪 Experimental comparison of 6 training configurations

📊 Accuracy up to 95.21% on the test set

📉 Early stopping, confusion matrix visualization, and OneCycleLR for stable training

# 🛠️ Stack

Python, PyTorch, torchvision

NumPy, scikit-learn, matplotlib, seaborn

CLI integration via argparse


# 📊 Results (Test Accuracy)

Baseline 93.0

Weighted Sampler 93.4

Mixup (α=0.4) 94.8

CutMix (α=1.0) 94.3

Mixup + Weighted Sampler 95.21

CutMix + Weighted Sampler 94.6


