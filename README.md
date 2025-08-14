# 🧪 Sandbox Science: Trying Big AI Ideas on Tiny Datasets

Welcome to the AI Summer School workshop on **Sandbox Science**! This session will teach you how to prototype advanced AI models using small, manageable datasets. You'll learn to test big ideas quickly without needing massive computational resources.

## 🎯 Workshop Overview

In this hands-on workshop, you'll:
- Build and train a **Joint Variational Autoencoder (JointVAE)** from scratch [[Dupont, 2018]](https://arxiv.org/abs/1804.00104)
- Work with two different toy datasets: **MNIST digits** and **Olivetti faces**
- Learn how to preprocess and prepare datasets for ML experiments
- Visualize latent space representations and model reconstructions
- Understand the power of prototyping with small datasets

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Familiarity with Jupyter notebooks

### Installation

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd Sandbox-Science
   ```

2. **Install dependencies using uv (recommended):**
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -e .
   ```


## 🧠 What is a JointVAE?

A **Joint Variational Autoencoder** is an extension of the standard VAE that learns both:
- **Continuous latent variables**: For smooth variations (like rotation, lighting)
- **Discrete latent variables**: For categorical features (like digit class, face identity)

This allows the model to disentangle different types of variations in your data!

### Architecture Overview
```
Input Image → Encoder → [Continuous + Discrete Latents] → Decoder → Reconstructed Image
     ↓                                                                      ↑
  64×64×1                    μ, σ + Gumbel-Softmax                      64×64×1
```

## 📚 Workshop Structure

### Part 1: MNIST Digits (Easy Mode) 🔢
**Notebook:** `train_model.ipynb`

- **Dataset:** MNIST handwritten digits (28×28 pixels)
- **Source:** TorchVision (loads automatically)
- **Challenge:** Learn to generate and manipulate digit images
- **Latent Space:** 3 continuous + 1 discrete (10 classes)

**What you'll learn:**
- Loading datasets with TorchVision
- Setting up the JointVAE architecture
- Training loop and loss functions
- Generating new samples
- Latent space traversals

### Part 2: Olivetti Faces (Challenge Mode) 👥
**Notebooks:** `make_faces.ipynb` → `train_model_olvetti.ipynb`

- **Dataset:** Olivetti faces (64×64 pixels, 400 faces of 40 people)
- **Source:** Scikit-learn (requires preprocessing)
- **Challenge:** Learn facial feature disentanglement
- **Latent Space:** 2 continuous + 1 discrete (10 classes)

**What you'll learn:**
- Preprocessing data from scikit-learn
- Creating custom datasets from numpy arrays
- Working with grayscale face images
- Facial feature manipulation through latent space

## 📁 Project Structure

```
Sandbox-Science/
├── 📓 train_model.ipynb           # Main MNIST training notebook
├── 📓 train_model_olvetti.ipynb   # Olivetti faces training notebook  
├── 📓 make_faces.ipynb            # Face dataset preparation
├── 📓 load_model.ipynb            # Model loading and inference
│
├── 🧠 jointvae/                   # Core model implementation
│   ├── models.py                  # VAE architecture
│   └── training.py                # Training utilities
│
├── 🛠️ utils/                       # Helper functions
│   ├── dataloaders.py             # Dataset loading utilities
│   └── load_model.py              # Model persistence
│
├── 📊 viz/                        # Visualization tools
│   ├── visualize.py               # Main visualization class
│   └── latent_traversals.py       # Latent space exploration
│
├── 🎭 face_images/                # Preprocessed face dataset (100 faces)
└── 📋 pyproject.toml              # Project dependencies
```

## 🎯 Step-by-Step Guide

### 🥇 Start with MNIST

1. **Open `train_model.ipynb`**
2. **Run all cells** to see the complete workflow:
   - Data loading with TorchVision
   - Model architecture definition
   - Training for 2 epochs (quick demo)
   - Sample generation and latent traversals

3. **Key concepts to observe:**
   - How the reconstruction loss decreases
   - Generated samples improve over time
   - Latent traversals show smooth transitions

### 🥈 Move to Olivetti Faces

1. **First, run `make_faces.ipynb`** to prepare the dataset:
   - Loads Olivetti faces from scikit-learn
   - Saves 100 faces as individual `.npy` files
   - Creates the `face_images/` directory

2. **Then open `train_model_olvetti.ipynb`**:
   - Uses custom dataloader for face images
   - Smaller latent space (2D continuous)
   - Trains on facial features

### 🎨 Experiment and Explore!

- **Modify latent dimensions**: Try different continuous/discrete combinations
- **Adjust training epochs**: See how longer training affects quality
- **Change architectures**: Experiment with layer sizes
- **Visualize results**: Use the built-in visualization tools

## 🔧 Key Components

### Model Architecture (`jointvae/models.py`)
- **Encoder**: CNN layers that compress images to latent codes
- **Decoder**: Transposed CNN layers that reconstruct images
- **Latent Space**: Split into continuous (Gaussian) and discrete (Gumbel-Softmax) parts

### Training (`jointvae/training.py`)
- **β-VAE Loss**: Balances reconstruction vs. latent regularization
- **Capacity Scheduling**: Gradually increases latent capacity during training
- **Joint Training**: Handles both continuous and discrete latents

### Visualization (`viz/visualize.py`)
- **Reconstructions**: Compare original vs. reconstructed images
- **Samples**: Generate new images from random latent codes
- **Latent Traversals**: See how changing one latent dimension affects images

## 🎨 Visualization Examples

The workshop includes several visualization tools:

- **`viz.reconstructions()`**: Shows original vs. reconstructed images
- **`viz.samples()`**: Generates completely new samples
- **`viz.latent_traversal_line()`**: Shows effect of varying one latent dimension
- **`viz.all_latent_traversals()`**: Grid showing all latent dimensions

## 🚀 Next Steps & Extensions

After completing the workshop, try these challenges:

### 🌟 Beginner Extensions
- Train for more epochs and compare results
- Try different latent space dimensions
- Experiment with different batch sizes

### 🔥 Advanced Challenges
- Add your own dataset (cats, cars, etc.)
- Implement β-VAE with different β values
- Add conditional generation (class-specific sampling)
- Try different architectures (deeper networks, attention layers)

### 🎯 Research Directions
- Implement other disentanglement methods (β-TCVAE, Factor-VAE)
- Add adversarial training components
- Experiment with different posterior distributions
- Try semi-supervised learning with partially labeled data

## 🤝 Tips for Success

1. **Start Simple**: Begin with MNIST to understand the concepts
2. **Iterate Quickly**: Use small datasets to test ideas fast
3. **Visualize Everything**: The viz tools are your best friends
4. **Experiment Freely**: Try breaking things to learn how they work
5. **Think Small**: Small datasets can teach you big lessons about AI


## 🏆 Workshop Goals

By the end of this session, you should be able to:
- ✅ Understand the JointVAE architecture and its components
- ✅ Load and preprocess datasets from multiple sources
- ✅ Train a generative model from scratch
- ✅ Visualize and interpret latent representations
- ✅ Apply these techniques to your own toy datasets
- ✅ Appreciate the power of prototyping with small data

---

**Happy experimenting! 🧪✨**

*Remember: The best way to learn AI is by building and breaking things. This sandbox is your playground*
