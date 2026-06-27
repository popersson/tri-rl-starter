# Tri RL Starter

Lean starter package for triangular-mesh improvement experiments.

The package is intentionally separate from the quad/blocking code. It provides:

- Random geometries: `regular_polygon`, `random_polygon`, `polygon_hole`,
  `grid_orth`, `grid_oct`.
- Triangle/PSLG meshing through the Python `triangle` package.
- A pure triangular `TriMesh` data structure with `t2t`, `t2n`, vertex
  adjacency, boundary nodes, corner nodes, ideal degrees, and minimum degrees.
- Degree metrics: L1, L2, L2 squared, and Linf.
- Triangle mean-ratio quality metrics.
- Matplotlib plotting with degree-mismatch node coloring.

Run:

```bash
python -m tri_rl_starter.driver --case random_polygon --seed 1
```

If `--seed` is omitted, the driver chooses a fresh random seed and prints it.

## Setup

From this repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

The package can also be run directly from the repository root without
installation as long as the dependencies are available in the active Python
environment.

Refined quality meshing:

```bash
python -m tri_rl_starter.driver --case polygon_hole --seed 4 --hmax 0.15
```

Regular circle-like polygon:

```bash
python -m tri_rl_starter.driver --case regular_polygon --nvertices 64 --hmax 0.12
```

`regular_polygon` ignores `--seed`; `--nvertices` is the only geometry
parameter.

## IPython Workflow

From the repository root, the closest replacement for the old
`run driver.py --args...` workflow is:

```python
%run -m tri_rl_starter.driver --case polygon_hole --seed 4 --hmax 0.15
```

Direct script running also works:

```python
%run tri_rl_starter/driver.py --case polygon_hole --seed 4 --hmax 0.15
```

For exploratory work, importing the package is usually nicer:

```python
%load_ext autoreload
%autoreload 2

from tri_rl_starter.geometry import generate_case_mesh
from tri_rl_starter.plotting import plot_mesh

mesh = generate_case_mesh(case="grid_oct", seed=3, hmax=0.4)
mesh.summary()
plot_mesh(mesh)
```

Simple sample smoothing:

```python
smoothed = mesh.smoothed(iterations=10, omega=0.5)
plot_mesh(smoothed)
```

The smoother uniformly spaces non-corner boundary nodes along each polygon side
and applies a few graph-Laplacian steps to interior nodes.

`plot_mesh` reuses the current Matplotlib figure by default and calls
`show(block=False)`, so repeated calls should update the existing window in
IPython. If you want a blocking script-style plot, use:

```bash
python -m tri_rl_starter.driver --case random_polygon --plot-block
```

## Degree Colors

Degree-deviation plotting is discrete by default, with integer bins. Neutral
nodes are small black dots; nonzero nodes use distinct categorical colors. To
switch back to the smooth color scale:

```bash
python -m tri_rl_starter.driver --degree-colors continuous
```

To force a specific integer color range:

```bash
python -m tri_rl_starter.driver --degree-color-min -3 --degree-color-max 5
```

## Smoothing

The driver can apply the sample smoother before reporting and plotting:

```bash
python -m tri_rl_starter.driver --case polygon_hole --hmax 0.15 --smooth
```

It is deliberately simple and meant as a placeholder for later quality-based
optimization.

Notes:

- Triangle ideal degree uses 60 degrees as the target angle, so interior nodes
  have ideal degree 6 and straight boundary nodes have ideal degree 4.
- Boundary target degrees are rounded by default for now because that gives
  convenient integer deviation signals for RL rewards. The lower-level mesh
  constructor keeps this choice local.
- `hmax` is approximated with Triangle's maximum-area refinement option. This
  mimics the scalar part of the Julia `TriMeshGen.jl` helper, but not its
  callback-based spatial size function yet.

## GitHub

After creating an empty private repository named `tri-rl-starter` on GitHub:

```bash
git remote add origin git@github.com:YOUR_USER/tri-rl-starter.git
git push -u origin main
```
