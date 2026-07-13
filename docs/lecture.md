# `lecture.ipynb` — Micrograd From Scratch

> The canonical, end-to-end micrograd lecture notebook. Builds a scalar-valued autograd engine (`Value` class) from first principles, derives backpropagation via the chain rule, verifies gradients against PyTorch, then assembles a tiny MLP and trains it with manual gradient descent. This is the foundation of the entire "Zero to Hero" curriculum — everything that follows (makemore, nanoGPT) assumes you understand what happens inside this notebook.

## Lecture Reference
- **Video:** "The spelled-out intro to neural networks and backpropagation: building micrograd" (Andrej Karpathy, *Neural Networks: Zero to Hero* — Lecture 1).
- **Original notebook:** `micrograd/lecture.ipynb` (local, 97 KB, 19 cells, all code — no markdown cells).

## Status
**Complete.** Fully executed top to bottom with stored outputs. No exercises.

## Concepts Demonstrated
- Numerical differentiation via finite differences (`h = 1e-8`).
- The chain rule as the engine of backpropagation.
- Reverse-mode automatic differentiation over a dynamically-built DAG.
- Topological sort as the ordering primitive for backward passes.
- Local `_backward` closures capturing per-operation gradient rules.
- PyTorch's `autograd` as a ground-truth baseline for custom gradients.
- Manual SGD: `zero_grad → backward → p.data += -lr * p.grad`.
- MSE loss for binary classification.

## Structure (in execution order)
The notebook has **no markdown cells** — sectioning is implicit via code comments. Logical sections:

1. Imports (`math`, `random`, `numpy`, `matplotlib`).
2. Derivative intuition: scalar function `f(x) = 3x² − 4x + 5`, finite-difference estimate.
3. Multi-input expression `d = a*b + c` and its partial derivatives.
4. **The `Value` class** (the autograd engine).
5. Sanity tests of `Value` ops.
6. Graphviz `trace` / `draw_dot` helpers.
7. `tanh` activation plot.
8. Hand-built single neuron `o = tanh(x1·w1 + x2·w2 + b)`.
9. PyTorch gradient verification.
10. `Neuron` / `Layer` / `MLP` classes.
11. Toy dataset + manual gradient-descent training loop.
12. Final predictions.

## Detailed Walkthrough

### Derivative intuition (cells 0–6)
- Plots `f(x) = 3x² − 4x + 5`, then estimates `f'(x)` numerically with `h = 1e-8`. The "tiny h" demonstration establishes that the derivative is a sensitivity measure — the foundation for why gradient descent works.
- Extends to `d = a*b + c`, computing `dd/dc` by hand with finite differences to generalize to multi-input functions.

### The `Value` class (cell 7)
The autograd core. Each `Value` stores `data`, `grad`, a set of `_prev` children, the `_op` that produced it, and a `_backward` closure (initially a no-op). Operations:

- **`__add__`** — `out = a + b`; backward: `a.grad += out.grad`, `b.grad += out.grad`.
- **`__mul__`** — `out = a * b`; backward: `a.grad += b.data * out.grad`, `b.grad += a.data * out.grad`.
- **`__pow__`** — `out = a ** k` (k int/float); backward: `a.grad += k * a^(k-1) * out.grad`.
- **`relu`** — `out = max(0, a)`; backward: `a.grad += (out.data > 0) * out.grad`.

Derived ops built on these primitives: `__neg__`, `__sub__`, `__radd__`, `__rsub__`, `__rmul__`, `__truediv__`, `__rtruediv__`. Note this version uses `tanh` and `exp` (the lecture variant) rather than the `relu`-only version in `micrograd/engine.py`.

**`backward()`** — builds a topological sort of the DAG rooted at `self`, sets `self.grad = 1`, then walks `reversed(topo)` calling each node's `_backward()`. The topological order guarantees every node's `grad` is fully accumulated before its own `_backward` runs.

### Graph visualization (cells 9–10)
`trace(root)` recursively collects all nodes/edges; `draw_dot(root)` builds a Graphviz `Digraph` with record-style nodes showing `{label | data | grad}` plus op nodes. This is the tool used throughout to visualize the forward graph and the backward gradient flow.

### Hand-built neuron (cell 11)
Constructs `o = tanh(x1·w1 + x2·w2 + b)` by hand with explicit `Value` objects. Notably, `tanh` is broken into its constituent ops `(e-1)/(e+1)` via `exp` rather than called as `.tanh()` — the direct `.tanh()` call is present but commented out. After `o.backward()`, `draw_dot(o)` renders the full computation graph with gradients populated on every node.

### PyTorch verification (cells 12–13)
Re-implements the same neuron in PyTorch with `torch.Tensor([…]).double()` and `requires_grad=True`. Runs `y.backward()` and asserts the forward value and the input gradients match the custom micrograd result. This is the lecture's demonstration that the hand-rolled engine is numerically identical to a production autograd.

### `Neuron` / `Layer` / `MLP` (cell 14)
- `Neuron(nin, nonlin=True)` — `w` is a list of `Value`s, `b` is a `Value(0)`; `__call__` computes `act = sum(wi*xi) + b` then `act.relu()` (or `act` if linear).
- `Layer(nin, nout)` — a list of `Neuron`s; `__call__` returns a list (or a single `Value` if `nout == 1`).
- `MLP(nin, nouts)` — stacks `Layer`s, with `nonlin=True` on all layers except the last.
- `parameters()` flattens all `w` and `b` across the entire model.

### Training (cells 15–18)
- Instantiates `MLP(3, [4, 4, 1])` and runs it on `[2.0, 3.0, -1.0]`.
- Toy dataset: 4 examples `xs` with binary targets `ys = [1.0, -1.0, -1.0, 1.0]`.
- **Manual gradient descent loop** (20 steps):
  - Forward: `ypred = [model(x) for x in xs]`.
  - Loss: `loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))` (MSE).
  - `model.zero_grad()` → `loss.backward()` → `p.data += -0.01 * p.grad`.
  - Prints loss each step.

## Key Results
- **PyTorch gradient verification** (cell 13): exact match — `x1.grad = -1.5`, `w1.grad = 1.0`.
- **Training loss**: `3.187` → `0.183` over 20 steps.
- **Final predictions** (cell 18): `[0.877, -0.821, -0.715, 0.764]` — correctly matching targets `[1, -1, -1, 1]` in sign and magnitude.
- Graphviz computation-graph plot of the hand-built neuron with all gradients populated.

## Caveats / Notes
- The `Value` class in this notebook differs slightly from `micrograd/micrograd/engine.py`: the notebook uses `exp`/`tanh` while the packaged engine uses `relu`. Both are valid; `tanh` is what the lecture video uses.
- No markdown cells — the notebook is code-only with implicit sectioning. This matches Karpathy's live-coding style.
- The `Value` class here includes `exp` and `tanh` as explicit primitives (not derived from `__pow__`), which is why the neuron's `tanh` is broken into `(e-1)/(e+1)` in cell 11 — to exercise the `exp` primitive.

## How to Run
```bash
cd micrograd
jupyter notebook lecture.ipynb
# or, headless:
jupyter nbconvert --to notebook --execute lecture.ipynb --output lecture.ipynb
```
Dependencies: `math`, `random`, `numpy`, `matplotlib`, `graphviz` (Python package + system binary), `torch` (for the verification cell only).
