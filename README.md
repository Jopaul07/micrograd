# Micrograd — The Autograd Engine

> **Chapter 1 of Andrej Karpathy's *Neural Networks: Zero to Hero* curriculum.** A tiny scalar-valued autograd engine (~100 LOC) and a PyTorch-like neural-network library (~50 LOC) built on top of it. The DAG operates over scalars only — each neuron is chopped into its individual adds and multiplies — yet this is enough to build and train entire deep neural nets. This is the foundation of the entire curriculum: makemore and nanoGPT both assume you understand what happens inside `micrograd/`.

Inspired by Andrej Karpathy's *Zero to Hero* curriculum, this project bypasses high-level abstractions to construct the core mathematical building blocks of deep learning from the ground up.

---

## Lecture Reference
- **Video:** "The spelled-out intro to neural networks and backpropagation: building micrograd" (Andrej Karpathy, *Neural Networks: Zero to Hero* — Lecture 1).

---

## Directory Contents

| File | Type | Description | Status |
|---|---|---|---|
| `micrograd/engine.py` | Source (94 LOC) | The `Value` class — scalar autograd engine. `+`, `*`, `**`, `relu`, `pow`, derived ops; topological-sort `backward()`. | Complete |
| `micrograd/nn.py` | Source (60 LOC) | `Module`/`Neuron`/`Layer`/`MLP` — PyTorch-like API on `Value`. | Complete |
| `micrograd/__init__.py` | Source | Package marker (empty). | Complete |
| `test/test_engine.py` | Test | Cross-validates micrograd gradients against PyTorch autograd. | Complete, passing |
| `lecture.ipynb` | Notebook (97 KB) | Full from-scratch build of `Value`, `Neuron`, `MLP`; manual SGD training. | Complete |
| `demo.ipynb` | Notebook (70 KB) | Packaged micrograd on `make_moons`; SVM hinge + L2; 100% accuracy. | Complete |
| `exercises_micrograd.ipynb` | Notebook (15 KB) | Analytical/numerical derivatives + softmax/NLL autograd; PyTorch-verified. | Complete (all solved) |
| `practice_book1.ipynb` | Notebook (7.7 KB) | From-memory re-implementation scratchpad. | **In-progress** (partial) |
| `trace_graph.ipynb` | Notebook (3.2 KB) | Graphviz `trace`/`draw_dot` utilities. | Complete (unexecuted) |
| `docs/` | Folder | Per-notebook chapter READMEs (see below). | Complete |

---

## Core Concepts Demonstrated

1. **Forward Pass Graph Construction:** Stacking basic algebraic operations (`+`, `*`, `**`, `relu`, `tanh`) while tracking parental node linkages.
2. **The Chain Rule in Code:** Storing a local `_backward` closure for each operation and calling it via topological sorting to guarantee correct gradient ordering.
3. **Reverse-Mode Automatic Differentiation:** Recursive backward pass across a dynamically-built DAG.
4. **Neural Network Modules:** `Neuron` → `Layer` → `MLP` composition with a PyTorch-like `parameters()` API.
5. **Optimization Loop:** Initializing a model, computing loss (MSE or hinge), zeroing gradients, executing manual SGD step updates.
6. **Gradient Verification:** Cross-validating custom gradient outputs against PyTorch's `autograd` engine to ensure numerical precision.

---

## Chapter Walkthrough Index

| # | Notebook | Topic | Status |
|---|---|---|---|
| 1 | [`docs/lecture.md`](docs/lecture.md) | Full from-scratch build of the autograd engine + MLP training. | Complete |
| 2 | [`docs/demo.md`](docs/demo.md) | Packaged micrograd on `make_moons`; decision-boundary viz. | Complete |
| 3 | [`docs/exercises_micrograd.md`](docs/exercises_micrograd.md) | Derivatives + softmax/NLL autograd exercises. | Complete (all solved) |
| 4 | [`docs/practice_book1.md`](docs/practice_book1.md) | From-memory re-implementation scratchpad. | **In-progress** |
| 5 | [`docs/trace_graph.md`](docs/trace_graph.md) | Graphviz trace/draw_dot utilities. | Complete (unexecuted) |

---

## Cross-Notebook Results Summary

| Notebook | Model / Task | Key Result |
|---|---|---|
| `lecture.ipynb` | `MLP(3,[4,4,1])` on 4-point toy set | Loss `3.187 → 0.183`; predictions match targets. |
| `demo.ipynb` | `MLP(2,[16,16,1])` on `make_moons` (337 params) | Loss `0.896 → 0.011`; accuracy `50% → 100%`. |
| `exercises_micrograd.ipynb` | Derivatives + softmax/NLL | All checks `OK`; PyTorch-verified to 1e-5. |
| `practice_book1.ipynb` | Re-implementation from memory | `Value` class done; `MLP` and training loop missing. |
| `trace_graph.ipynb` | Graphviz utilities | No stored outputs (unexecuted). |

---

## Prerequisites & Core Dependencies

The codebase runs on **Python 3.x** and relies on a focused stack:

- `torch` — Used strictly as a ground-truth baseline to verify custom gradient calculations.
- `graphviz` — Used to generate and render visual representations of the computation graphs (Python package + system binary).
- `numpy` — For vectorized evaluation and data structures.
- `matplotlib` — For plotting training loss curves and decision boundaries.

---

## How to Run

### Run the test suite
```bash
cd micrograd
pip install torch numpy graphviz matplotlib
python -m pytest test/test_engine.py -v
```

### Run a notebook
```bash
cd micrograd
jupyter notebook lecture.ipynb
# or headless:
jupyter nbconvert --to notebook --execute lecture.ipynb --output lecture.ipynb
```

### Use the packaged library
```python
from micrograd.engine import Value
from micrograd.nn import MLP

model = MLP(2, [16, 16, 1])
# ...forward, loss, backward, step...
```
Run from the `micrograd/` directory or `pip install -e .` to make the package importable globally.

---

## Status & Caveats

- **`practice_book1.ipynb` is the only in-progress notebook.** The `Value` class is complete, but `MLP`, the training loop, the dataset, and the loss function are all missing. Natural next steps: add `relu`/`__rtruediv__` to `Value`, implement `MLP`, wire `parameters()` as a method, add a toy training loop.
- **`trace_graph.ipynb` has no stored outputs** — re-run it to see the Graphviz renderings (requires the system Graphviz binary).
- **`lecture.ipynb` vs. `micrograd/engine.py`:** The notebook uses `exp`/`tanh` as primitives; the packaged engine uses `relu`. Both are valid; `tanh` matches the lecture video.
- All gradient checks pass; all training loops have been executed with stored outputs.
- The codebase is a clean, faithful study implementation — not extended beyond Karpathy's original yet.

---

## Acknowledgments
- **Andrej Karpathy:** For the exceptional `micrograd` video lecture series and educational framework.
