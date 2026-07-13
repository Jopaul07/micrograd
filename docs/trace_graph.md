# `trace_graph.ipynb` — Graphviz Trace Utility

> A small utility notebook for visualizing the micrograd computation graph with Graphviz. Imports the packaged `micrograd` `Value` (and `nn`) and uses `trace`/`draw_dot` to render the autograd DAG for two toy examples. Reusable snippet for graph tracing rather than a tutorial.

## Lecture Reference
- **Video:** Companion to "The spelled-out intro to neural networks and backpropagation: building micrograd" (Andrej Karpathy, *Neural Networks: Zero to Hero* — Lecture 1).
- **Original notebook:** `micrograd/trace_graph.ipynb` (local, 3.2 KB, 6 cells).

## Status
**Complete as a utility, but unexecuted.** None of the cells have stored outputs — the graph renderings were either cleared or never run/saved in this version. The code is correct and runnable.

## Concepts Demonstrated
- `trace(root)` — recursively walks the DAG from `root`, collecting all nodes and edges.
- `draw_dot(root)` — builds a Graphviz `Digraph` with record-style nodes showing `{label | data | grad}` and op nodes.
- Layout options: `rankdir='LR'` (left-to-right, default) or `'TB'` (top-to-bottom).
- Rendering the graph to a file via `dot.render(...)`.

## Structure (in execution order)
No markdown cells — six code cells only:
1. Cell 0 — import `Digraph` from `graphviz`.
2. Cell 1 — import `Value` from `micrograd.engine`.
3. Cell 2 — `trace(root)` and `draw_dot(root, format='svg', rankdir='LR')` definitions.
4. Cell 3 — toy example: `x = Value(1.0)`, `y = (x*2 + 1).relu()`, `y.backward()`, `draw_dot(y)`.
5. Cell 4 — 2D neuron example: `n = nn.Neuron(2)`, `x = [Value(1.0), Value(-2.0)]`, `y = n(x)`, `y.backward()`, `draw_dot(y)`.
6. Cell 5 — `dot.render('gout')` — renders the last graph to a file named `gout`.

## Detailed Walkthrough

### `trace` and `draw_dot` (cell 2)
- `trace(root)`: recursively builds a set of all nodes reachable from `root` and a set of all edges `(parent, child)`. This is the standard DAG traversal used to enumerate the computation graph.
- `draw_dot(root, format='svg', rankdir='LR')`: calls `trace(root)`, then constructs a `graphviz.Digraph` with:
  - Record-style nodes: `{label | data | grad}` for `Value` nodes, op-name nodes for operations.
  - Edges from children to parents.
  - `rankdir` controls layout direction (`'LR'` = left-to-right, `'TB'` = top-to-bottom).
  - Returns the `Digraph` object (which Jupyter renders inline if it's the last expression in a cell).

### Toy example (cell 3)
`x = Value(1.0)`, `y = (x*2 + 1).relu()`, `y.backward()`, `draw_dot(y)`. A minimal graph: `x → (x*2) → (+1) → relu → y`, with `x.grad = 2.0` (the chain-rule result through the relu and the multiply). Note: no output is shown in the notebook for this cell — the `draw_dot(y)` return value isn't displayed (likely because it wasn't the last expression or the cell wasn't run).

### 2D neuron example (cell 4)
`n = nn.Neuron(2)`, `x = [Value(1.0), Value(-2.0)]`, `y = n(x)`, `y.backward()`, `dot = draw_dot(y)`. Renders the full neuron computation graph: two weight-multiplies, a bias-add, a sum, and a relu. The last line `dot` would render the graph inline.

### Render to file (cell 5)
`dot.render('gout')` — saves the last graph to a file named `gout` (e.g. `gout.svg` or `gout.pdf`, depending on the `format` argument). This is the mechanism for persisting a graph rendering outside the notebook.

## Key Results
None stored in the notebook — all cells have `n_out=0` (no outputs). The expected runtime outputs are Graphviz SVG graphs of the two computation graphs and a saved `gout` file.

## Caveats / Notes
- **No stored outputs.** To see the graphs, re-run the notebook (requires Graphviz installed system-wide: `brew install graphviz` on macOS, or `apt-get install graphviz` on Linux).
- This notebook is a **utility**, not a tutorial. It exists to be copy-pasted into other notebooks or scripts when you need to visualize a micrograd computation graph.
- The `draw_dot` function here is a standalone copy; `lecture.ipynb` has its own equivalent version. The packaged `micrograd` library does not ship a `draw_dot` helper — it's left to the user to define.
- `format='svg'` is the default; use `format='png'` for raster output or `format='pdf'` for print.

## How to Run
```bash
cd micrograd
jupyter notebook trace_graph.ipynb
```
Dependencies: `graphviz` (Python package: `pip install graphviz`) **and** the Graphviz system binary (`brew install graphviz` on macOS). The notebook also imports from the local `micrograd` package, so run from the `micrograd/` directory or install it with `pip install -e .`.
