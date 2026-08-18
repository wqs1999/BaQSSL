# BaQSSL
# BaQSSL: Channel-Balanced Quantum Self-Supervised Learning with Adaptive Style Perturbation

This repository provides the official implementation of the paper:

**"BaQSSL: Channel-Balanced Quantum Self-Supervised Learning with Adaptive Style Perturbation"**

The repository is mainly developed to facilitate reproduction of the experimental results reported in the paper and to provide an implementation of the proposed quantum self-supervised learning framework.

## Overview

Quantum self-supervised learning (QSSL) provides a promising approach for learning representations from unlabeled data by combining parameterized quantum circuits with self-supervised objectives. However, existing QSSL methods may underutilize discriminative information across quantum feature channels and remain sensitive to style-related variations.

To address these limitations, BaQSSL introduces two key components:

- **Quantum Channel Balancing (QCB):**  
  Learns adaptive channel adjustment directions and strengths from quantum feature statistics to improve the utilization of discriminative information across quantum feature channels.

- **Quantum Style Perturbation (QSP):**  
  Generates diverse quantum feature perturbations with adaptive strengths to reduce sensitivity to style-related variations and improve feature robustness.

The proposed framework is evaluated on MNIST, FashionMNIST, and CIFAR-10 datasets under different classification settings.

## Requirements

The implementation is based on Python and PyTorch.

Recommended environment:

- Python >= 3.10
- PyTorch
- PennyLane
- torchvision
- scikit-learn
- numpy
