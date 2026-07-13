# `exercises_micrograd.ipynb` — Micrograd Exercises

> The official micrograd exercise notebook. Two sections: (1) analytical and numerical differentiation of a scalar function, and (2) extending a stripped-down `Value` class to support `exp`, `log`, division, etc., then implementing **softmax + negative-log-likelihood loss** with manual backprop, finally verified against PyTorch. Reinforces the lecture's core ideas through hands-on reimplementation.

## Lecture Reference
- **Video:** Companion to "The spelled-out intro to neural networks and backpropagation: building micrograd" (Andrej Karpathy, *Neural Networks: Zero to Hero* — Lecture 1).
- **Original notebook:** `micrograd/exercises_micrograd.ipynb` (local, 15 KB, 10 cells).

## Status
**Complete — all exercises solved and verified.** Every check prints `OK`.

## Concepts Demonstrated
- Analytical differentiation (hand-derived partial derivatives).
- One-sided numerical gradient: `(f(x+h) - f(x)) / h`.
- Central (symmetric) difference: `(f(x+h) - f(x-h)) / (2h)` — better accuracy at the same `h`.
- Extending the `Value` autograd class with new primitives (`exp`, `log`, `__truediv__`).
- Softmax as a composition of `exp`, sum, and division.
- Negative log-likelihood (NLL) loss via `log`.
- PyTorch autograd as ground truth for custom gradient verification.

## Structure (in execution order)
1. `# micrograd exercises` — "The following sections will test the knowledge and revise the core concepts learned!" (cell 0).
2. `## Section 1: derivatives` (cell 1).
3. Test function `f(a, b, c) = -a³ + sin(3b) − 1.0/c + b^2.5 − a^0.5`.
4. Exercise 1a — analytical gradient.
5. Exercise 1b — one-sided numerical gradient.
6. Exercise 1c — central-difference numerical gradient.
7. `## section 2: support for softmax` (cell 6).
8. Starter `Value` class (only `__add__` and `backward` filled in; student implements the rest).
9. Exercise 2 — softmax + NLL loss with manual backprop.
10. PyTorch verification.

## Detailed Walkthrough

### Section 1: Derivatives

**Test function** (cell 2): `f(a, b, c) = -a**3 + sin(3*b) - 1.0/c + b**2.5 - a**0.5`; verifies `f(2, 3, 4) = 6.336`.

**Exercise 1a — analytical gradient** (cell 3): `gradf(a, b, c)` returns `[df/da, df/db, df/dc]` computed by hand from calculus:
- `df/da = -3a² - 0.5 * a^(-0.5)`
- `df/db = 3cos(3b) + 2.5 * b^1.5`
- `df/dc = 1.0/c²`

Checked against `ans = [-12.354, 10.257, 0.0625]`.

**Exercise 1b — one-sided numerical gradient** (cell 4): `compute_gradient(...)` uses `h = 1e-8` and the `(f(x+h) - f(x))/h` approximation from the lecture video. Each partial is estimated by nudging one variable at a time.

**Exercise 1c — central difference** (cell 5): `compute_sym_gradient(...)` uses `(f(x+h) - f(x-h)) / (2h)`, demonstrating better accuracy at the same `h` — the `O(h²)` truncation error vs. the one-sided `O(h)` error.

All three methods print `OK` for every dimension against the analytical answer.

### Section 2: Softmax + NLL

**Starter `Value` class** (cell 7): A stripped-down version with only `__add__` and `backward` filled in. The student re-implements `__mul__`, `__pow__`, `exp`, `log`, `__truediv__`, `__rmul__`, `__radd__`, `__neg__`. All are filled in here (solved state).

**Exercise 2 — softmax + NLL** (cell 8):
- `softmax(logits)`: `exp` each logit, sum them, divide each by the sum. (Contains leftover debug `print` statements showing `counts`, `denominator`, `out` — these match the expected results.)
- `loss = -probs[3].log()` with the label being dimension 3 (the "correct" class).
- Calls `loss.backward()` and checks the four logit gradients against `ans = [0.0418, 0.839, 0.00565, -0.886]`.

The key insight: the softmax + NLL backward simplifies to `dlogits = probs - one_hot(y)`, which is exactly the closed-form cross-entropy gradient that appears later in `makemore/build_makemore_part4_backprop.ipynb`.

**PyTorch verification** (cell 9): Rebuilds the same softmax + NLL loss in `torch` and compares gradients. All checks `OK` (matching to within 1e-5).

## Key Results
- **Section 1**: all three derivative methods (analytical, one-sided numerical, central difference) print `OK` for every dimension.
- **Section 2**: `loss.data = 2.1755`; all four logit gradients match `ans` (all `OK`).
- **PyTorch verification**: all gradient checks `OK` (matching to within 1e-5).

## Caveats / Notes
- The softmax implementation contains leftover debug `print` statements (showing `counts`, `denominator`, `out`). These are not bugs — they print intermediate values that match the expected results, but they're noise in the output.
- The central-difference exercise (1c) is the most accurate numerical method at the same `h` — this is the motivation for why PyTorch's `gradcheck` uses central differences rather than one-sided.
- The softmax + NLL backward is a sneak preview of the closed-form cross-entropy gradient derived in detail in `makemore/build_makemore_part4_backprop.ipynb` (Exercise 2 there).
- This is the only micrograd notebook with a markdown section structure (`#` and `##` headings).

## How to Run
```bash
cd micrograd
jupyter notebook exercises_micrograd.ipynb
# or headless:
jupyter nbconvert --to notebook --execute exercises_micrograd.ipynb --output exercises_micrograd.ipynb
```
Dependencies: `math`, `numpy`, `torch` (for the verification cell).
