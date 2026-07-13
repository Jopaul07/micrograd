# `demo.ipynb` — Micrograd on `make_moons`

> End-to-end demo of the *packaged* micrograd library (imported from `micrograd.engine` / `micrograd.nn`) training an MLP classifier on the `make_moons` dataset — a binary 2D classification problem with a non-linear decision boundary. Demonstrates that the 150-line micrograd engine is sufficient to train a real (if small) neural network to 100% accuracy on a non-trivially-separable dataset.

## Lecture Reference
- **Video:** Companion to "The spelled-out intro to neural networks and backpropagation: building micrograd" (Andrej Karpathy, *Neural Networks: Zero to Hero* — Lecture 1).
- **Original notebook:** `micrograd/demo.ipynb` (local, 70 KB, 10 cells).

## Status
**Complete.** Fully executed with stored outputs (plots, loss/accuracy trajectory). No exercises.

## Concepts Demonstrated
- Using the packaged `micrograd` library as an external dependency (rather than the inline `Value` class from `lecture.ipynb`).
- `make_moons` dataset construction and visualization.
- SVM max-margin hinge loss: `(1 + -yi * scorei).relu()` — an alternative to MSE that penalizes only confident wrong predictions.
- L2 regularization: `alpha * sum(p**2 for p in params)`.
- Inline mini-batch dataloader (random subsample each step).
- Learning-rate decay schedule: `lr = 1.0 - 0.9 * k / steps`.
- Decision-boundary visualization on a meshgrid.

## Structure (in execution order)
1. `### MicroGrad demo` (the only markdown heading, cell 0).
2. Imports: numpy, matplotlib, `Value`/`Neuron`/`Layer`/`MLP` from the local `micrograd` package.
3. Seeding: `np.random.seed(1337)`, `random.seed(1337)`.
4. `make_moons` dataset construction (100 samples, noise=0.1, y mapped to {−1, +1}).
5. Scatter plot of the dataset.
6. Model instantiation: `MLP(2, [16, 16, 1])`.
7. `loss(batch_size=None)` function — hinge loss + L2 regularization + accuracy.
8. Training loop (100 steps of SGD).
9. Decision-boundary visualization on a meshgrid.
10. Trailing empty cell.

## Detailed Walkthrough

### Dataset (cells 4–5)
Builds `make_moons(n_samples=100, noise=0.1)` from sklearn (or hand-rolled equivalent), maps the `y` labels from `{0, 1}` to `{-1, +1}` (the convention required by the hinge loss), and scatter-plots the two interleaving half-circles — the canonical non-linearly-separable 2D classification benchmark.

### Model (cell 5)
`model = MLP(2, [16, 16, 1])` — two hidden layers of 16 neurons (ReLU), single linear output. Prints the structure and parameter count: **337 parameters**.

### Loss function (cell 6)
`loss(batch_size=None)`:
- If `batch_size` is given, subsamples a random mini-batch from the full dataset; otherwise uses all 100 examples.
- Forward pass: `scores = [model(x) for x in X]`.
- **Hinge loss**: `sum((1 + -yi * scorei).relu() for yi, scorei in zip(y, scores))` — penalizes predictions that are either on the wrong side of the decision boundary or on the correct side but within the margin of 1.
- **L2 regularization**: `alpha * sum(p**2 for p in model.parameters())` with `alpha = 1e-4`.
- **Accuracy**: fraction of examples where `scorei.data > 0` matches `yi > 0`.
- Returns `(total_loss, accuracy)`.

### Training loop (cell 7)
100 steps of manual SGD:
- `lr = 1.0 - 0.9 * k / 100` — linear decay from 1.0 to 0.1.
- Forward: `total_loss, acc = loss(batch_size=...)` (uses the inline mini-batch).
- `model.zero_grad()` → `total_loss.backward()` → `p.data -= lr * p.grad`.
- Prints loss and accuracy each step.

### Decision boundary (cell 8)
Builds a meshgrid over the 2D input space (`h = 0.25` spacing), forwards each grid point through the trained model, thresholds the score at 0, and renders a colored contour map overlaying the data points. This visualizes the learned non-linear decision boundary that separates the two moon classes.

## Key Results
- **Initial loss**: `0.896`, accuracy `50%` (chance).
- **Final (step 99) loss**: `0.01098`, accuracy `100%`.
- Two PNG plots stored: the moon-shaped data scatter, and the colored decision-boundary contour map showing clean separation of the two classes.

## Caveats / Notes
- The loss here is hinge + L2, not MSE. This is intentional — Karpathy's demo uses hinge loss to illustrate that micrograd supports arbitrary loss functions, not just the MSE from `lecture.ipynb`.
- The learning-rate schedule is linear decay, not the exponential/step decay used in the makemore/nanoGPT chapters. The decay matters: without it, the final loss plateaus higher because late-step updates are too large.
- The `batch_size` argument to `loss()` enables mini-batch SGD, but the default `None` uses the full 100-example dataset. The training loop passes a `batch_size` to subsample.
- No train/test split — this is a demonstration of the engine, not a generalization study. The `make_moons` problem is small enough that 100% train accuracy is the goal.

## How to Run
```bash
cd micrograd
jupyter notebook demo.ipynb
# or headless:
jupyter nbconvert --to notebook --execute demo.ipynb --output demo.ipynb
```
Dependencies: `numpy`, `matplotlib`, `graphviz` (for the packaged `micrograd` to render graphs, though not used here), and the local `micrograd` package (must be importable — run from the `micrograd/` directory or install with `pip install -e .`).
