# GraphWGAN for Molecular Generation (QM9)

This repository implements a **graph-based Wasserstein GAN with gradient penalty (WGAN-GP)** to generate small organic molecules from the **QM9** dataset.

Molecules are represented as **multi-relational graphs** (atom features + bond-type adjacency). A **graph generator** maps latent vectors to molecular graphs, and a **relational graph convolutional discriminator** scores their realism. Generated graphs are converted back to RDKit molecules and SMILES for analysis and visualization.

---

## ✨ Key Features

- **Graph representation of molecules**
  - Atoms: C, N, O, F (+ padding) as one-hot vectors.
  - Bonds: SINGLE, DOUBLE, TRIPLE, AROMATIC (+ no-bond channel).
  - Fixed-size tensors:
    - Adjacency: `(BOND_DIM, NUM_ATOMS, NUM_ATOMS)`
    - Features: `(NUM_ATOMS, ATOM_DIM)`

- **Relational Graph Convolutional Discriminator**
  - Custom `RelationalGraphConvLayer` that:
    - Aggregates neighbor features per bond type.
    - Applies relation-specific weight matrices.
    - Sums over relations and applies nonlinearity.
  - Global average pooling → graph-level embedding → MLP → scalar critic score.

- **Graph Generator**
  - MLP trunk on latent vector `z` (e.g. `[128, 256, 512]` with `tanh` + dropout).
  - Two output heads:
    - Adjacency head: produces symmetric bond-type logits, softmax over bond channels.
    - Feature head: produces atom-type logits, softmax over feature channels.
  - Outputs **soft** molecular graphs that can be discretized via `argmax`.

- **Wasserstein GAN with Gradient Penalty (WGAN-GP)**
  - Critic loss:  
    \[
    L_D = \mathbb{E}[D(\text{fake})] - \mathbb{E}[D(\text{real})] + \lambda \cdot \text{GP}
    \]
  - Generator loss:  
    \[
    L_G = -\mathbb{E}[D(\text{fake})]
    \]
  - Gradient penalty computed on interpolations between real and generated graphs, for both adjacency and features.

- **End-to-end pipeline**
  - SMILES → RDKit → graph tensors for training.
  - Latent → generator → graph → RDKit → SMILES for generation.
  - Save generated SMILES to CSV and visualize molecules in a grid.

---

## 🧱 Project Structure

```text
.
├── Train_model.ipynb         # Training WGAN-GP on QM9 graph data
├── Generate_molecules.ipynb  # Sampling new molecules from the trained generator
├── my_model/                 # (Created after training)
│   ├── generator.weights.h5  # Saved generator weights
│   ├── discriminator.weights.h5  # (optional, if saved)
│   └── generated_smiles.csv  # Generated SMILES (from generation notebook)
└── qm9.csv                   # QM9 subset with SMILES (not included in repo)
