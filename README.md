# Micrograd: Scalar-Valued Autograd Engine & Neural Networks 🧠

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

Inspired by Andrej Karpathy's _Zero to Hero_ curriculum, this project bypasses high-level abstractions to construct the core mathematical building blocks of deep learning from the ground up.

---

## 🚀 Technical Highlights

- **Custom Autograd Engine:** Implements a scalar-valued `Value` class that dynamically constructs a Directed Acyclic Graph (DAG) during the forward pass.
- **Automated Backpropagation:** Executes recursive reverse-mode automatic differentiation across the computational graph using custom-mapped chain rule operations (`_backward`).
- **Neural Network Module:** Builds fundamental components including single `Neuron` nodes, fully connected `Layer` matrices, and Multi-Layer Perceptrons (MLPs).
- **Verification:** Cross-validates custom gradient outputs against PyTorch’s autograd engine to ensure absolute numerical precision.

---

## 🛠️ Prerequisites & Core Dependencies

The codebase runs on **Python 3.x** and relies on a focused stack of mathematical and visualization libraries:

- `torch` — Used strictly as a ground-truth baseline to test and verify custom gradient calculations.
- `graphviz` — Used to generate and render visual representations of the underlying computation graphs.
- `numpy` — For vectorized evaluation and data structural operations.
- `matplotlib` — For tracking and plotting training loss convergence curves.


## 📊 Core Concepts Demonstrated

1.  **Forward Pass Graph Construction:** Stacking basic algebraic operations (`+`, `*`, `**`, `relu`, `tanh`) while tracking parental node linkages.
2.  **The Chain Rule in Code:** Storing a local `_backward` function for each operation and calling it via topological sorting to guarantee correct gradient ordering.
3.  **Optimization Loop:** Initializing a mini-MLP, computing loss (such as MSE or Max-Margin loss), zeroing out old gradients, and executing manual Stochastic Gradient Descent (SGD) step updates.

---

## 📜 Acknowledgments

- **Andrej Karpathy:** For the exceptional `micrograd` video lecture series and educational framework.
