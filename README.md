# Overview

Code for the paper "How to sketch a learning algorithm." See Section 5.2 for context.

## Requirements

The only requirement is `matplotlib`.

```bash
pip install matplotlib
```

You don't even need this if you don't want the PNG figures: Just comment out the relevant parts.

## Implementation Notes

This implementation of microgpt is taken directly from microgpt (https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), with three changes: the code is refactored and updated to handle higher-order forward-mode automatic differentiation with polynomial ring computations, `eps_adam` is inside of the `sqrt` and increased from 1e-8 to 1e-3, and we use GELU instead of ReLU gates.

## Code Organization

All of the code for training and forward-mode differentiation is in `src`:
- `microgpt.py` is Karpathy's code, with the modifications above;
- `polys.py` contains polynomial arithmetic;
- `autograd_real.py` and `autograd_poly.py` implement automatic differentiation with float and polynomial objects, respectively.

## Experiments

The `generate_figX.py` scripts perform the experiments, store the data in `data/figX/`, store the results for plotting in `results/`, and generate matplotlib png plots in `results/`. You can then use `figX_data_to_tikz.py` to get tikz code for latex.
