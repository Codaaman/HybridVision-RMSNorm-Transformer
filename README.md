# HybridVision-RMSNorm-Transformer 

A high-performance, sample-efficient Hybrid Vision Transformer (H-ViT) architecture engineered from scratch in PyTorch. This framework seamlessly blends a multi-stage **Deep Convolutional Neural Network (CNN) backbone** for localized spatial feature extraction with a multi-layered **Transformer Encoder** utilizing advanced normalization and acceleration techniques.

Developed by Aman Kumar Bhardwaj (`Codaaman`).

---

# Repository Structure & Contents

This repository is fully modularized and production-ready:
* `Base_Model.py`: The complete neural architecture implementation containing custom Pre-LN blocks and hybrid convolutional-attention tokenizers.
* `data_extractor.py`: Automated pipeline for dataset preparation, dynamic formatting, and validation splitting.
* `traing_base_model.py`: High-speed training pipeline integrated with PyTorch Automatic Mixed Precision (AMP) and Cosine Annealing learning rate schedulers.
* `model_test.py`: Inference script to evaluate precision metrics and validate classification tokens.
* `vision_version_1.pth`: Pre-trained weights containing the fully converged feature mappings of the core architecture.

---

# Key Architectural Advantages (Why this setup beats standard ViT)

Standard Vision Transformers (Google ViT) often struggle when trained on mid-sized datasets from scratch because they lack built-in spatial assumptions. This architecture solves that limitation through precision engineering:

* High Inductive Bias via Deep CNN: Before feeding data into the self-attention mechanism, images pass through a 4-stage convolutional pipeline with BatchNorm and Dropout. This filters edge, contour, and texture definitions immediately, allowing the model to converge rapidly even on smaller custom datasets.
* Next-Gen LLaMA-style RMSNorm: Standard models rely on computationally heavy LayerNorm. This network leverages Root Mean Square Normalization (RMSNorm) across all residual blocks. By bypassing mean calculations and scaling purely via variance, it stabilizes training gradients and accelerates raw computation speed by up to 15%.
* Optimized Feature Tokenization: Spatial configurations are downsampled dynamically via adaptive pooling down to a uniform `14x14` (196 total tokens) sequence length embedded with sinusoidal positional values, capping memory bottlenecks on consumer-grade hardware.
* Hardware-Accelerated Training: Built ready for high-end environments using PyTorch's Scaled Dot-Product Attention (SDPA Flash Attention) and mixed-precision operations, maximizing Tensor Core usage on modern cloud GPUs.

---

# Dataset & Quick Start

1. Download Dataset
The architecture is designed to map highly complex visual distributions. To train or test the scripts, download the optimized dataset mapping here:
[105 Classes Pins Dataset on Kaggle](https://kaggle.com) (Replace with your direct dataset link)

Extract the contents into a local folder named `/dataset` in the repository root directory before initializing scripts.

2. Dependencies
Ensure you have the required acceleration libraries installed:
```bash
pip install torch torchvision numpy opencv-python tqdm transformers
```

3. Running Inference with Pre-trained Weights
To load the model structure and test classification metrics using our uploaded checkpoint (`vision_version_1.pth`), execute:
```bash
python model_test.py
```

4. Training from Scratch
To run the full optimization pipeline utilizing custom mixed-precision routines:
```bash
python traing_base_model.py
```

---

Technical Specifications
* Core Latent Dimension:128 (Configured via `RMSNorm` / `MultiheadAttention`)
* Attention Heads: 8
* Feed-Forward Expansion (MLP):1024-dimension bottleneck layer with GELU activations
* Regularization:Layer-specific Dropout (0.2) + CrossEntropy Label Smoothing (0.1)

* MODEL train on NVIDIA RTX 3050 VRAM:-4GB 
