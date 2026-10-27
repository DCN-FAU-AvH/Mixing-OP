# Self-Supervised In-Context Operator Learning for Optimal Mixing in Incompressible Flows

This repository contains the PyTorch implementation accompanying the paper

> W. Hu, R. Lai, and Z. Li. *Self-Supervised In-Context Operator Learning for Optimal Mixing in Incompressible Flows.* arXiv:2610.25556, 2026.

A transformer maps two initial particle clouds, carrying fixed labels `+1` and `-1`, to time-dependent coefficients of a divergence-free velocity basis. Training differentiates through RK4 particle trajectories and minimizes the terminal squared `Ḣ^{-1}` mix-norm plus kinetic energy. No precomputed optimal controls are required. The code supports four partition families in each of two and three dimensions.

The main entry points are:

| Script | Purpose | Geometry | Section in paper |
|---|---|---|---|
| `train.py` | train one 2D family | unit square; 4 velocity modes | §4, §5.2 |
| `train3d.py` | train one 3D family | unit cube; 24 velocity modes | §4, §5.3 |
| `example/test_cross_all_2d.py` | four-family particle visualization | unit square | §5.2 |
| `example/test3d_cross_family.py` | four-family Eulerian visualization | unit cube | §5.3 figures; see note below |
| `example/compare_particle_pde.py` | particle / Eulerian verification | unit square | §5.4 |

## Installation

Use Python 3.10 or later. From this directory, install the dependencies in your preferred virtual environment:

```bash
python -m pip install -r requirements.txt
```

Training supports CPU and CUDA; the device is selected automatically and can be set with `--device cpu` or `--device cuda`. Full training runs are intended for a GPU; the manuscript runs used PyTorch 2.8.0 on NVIDIA A100 GPUs. FFmpeg is optional and is needed only for MP4 output.

## Usage

Run the following commands from `Code-mixing-OP/`. Training creates a separate directory such as `out/rotation_line_d1/` or `out/3d_rotation_plane_d2/`, containing:

- `train.log` — parameters, losses, and training/test diagnostics;
- `ckpt_<iteration>.pt` — periodic model and optimizer checkpoints;
- `ckpt_final.pt` — final model weights and training arguments.

The commands below explicitly set the manuscript's kinetic-energy weights: `2e-3` in 2D and `2e-2` in 3D. Both training scripts otherwise default to `1e-3`. The remaining training defaults match the manuscript's hyperparameter table. Checkpoints are not included inside this code folder; train a model first or supply a checkpoint from the parent project's `results/` directory.

**Reproducibility note.** The current manuscript's numerical images match files under `../results/2d/`, `../results/3d/`, and `../results/2d/eulerian_hires/`. The older 2D/3D checkpoints there use kinetic weights different from the current table. The main checkpoints in `../results/2d_final/` and `../results/3d_final/` match the table. The existing plotting scripts also have separate evaluation settings, described below. The common-reference, diagonal-free evaluator used for the spectral-cutoff ablation table is not included. This folder therefore contains the core implementation and visualization tools, but is not yet a complete reproduction package for every current figure and table.

### (1) Two-dimensional training and evaluation

```bash
python train.py --family rotation_line --lam_kin 2e-3

python -m example.test_cross_all_2d \
    --checkpoint out/rotation_line_d1/ckpt_final.pt \
    --out_dir out/rotation_line_d1/figures \
    --N_grid 50 --N_T 200 --K_mix 12
```

Choose `rotation_line`, `general_line`, `circle`, or `level_set`. Each family is trained separately. The example evaluates the mix-norm at the manuscript's training cutoff `12`; the plotting script's own default is `24`. It uses a fixed family row order and selects approximately balanced partitions from the training parameter domain, with an expanded circle-radius range. For evaluation on both training and held-out parameters, use `python -m example.test_multi`.

### (2) Three-dimensional training and evaluation

```bash
python -m example.tests3d_sanity
python train3d.py --family rotation_plane --lam_kin 2e-2

PYTHONHASHSEED=0 python -m example.test3d_cross_family \
    --checkpoint out/3d_rotation_plane_d2/ckpt_final.pt \
    --N_x 48 --N_T_sim 60
```

Choose `rotation_plane`, `general_plane`, `sphere`, or `level_set_3d`. The sanity checks verify the basis and mix-norm implementation. The cross-family script produces Eulerian density snapshots on a `48³` grid; it does not implement the `30³` particle-grid evaluation described in the manuscript. `PYTHONHASHSEED=0` fixes the script's family-dependent seed offsets. Use `python -m example.test3d` for RK4 particle evaluation of an individual family.

### (3) Independent Eulerian verification

```bash
python -m example.compare_particle_pde \
    --checkpoint out/rotation_line_d1/ckpt_final.pt \
    --out_dir out/rotation_line_d1/eulerian \
    --N_x 256 --n_particles 6000
```

This applies the learned control to a finite-volume transport solver and compares its scalar snapshots and mix-norm with the particle calculation. The requested particle count is rounded to a square lattice (`6000` gives `77 × 77 = 5929` particles). Supply an explicit checkpoint path as above, or use `--all --results_dir ../results/2d_final` to evaluate the four family checkpoints in an existing results directory.

## Repository layout

```text
Code-mixing-OP/
├── .gitignore                   # macOS metadata and Finder exclusions
├── readme.md
├── requirements.txt
├── train.py                      # 2D RK4 rollout and training
├── train3d.py                    # 3D RK4 rollout and training
├── model/
│   ├── __init__.py              # exports MixingOperatorNet
│   ├── network.py               # shared two-cloud transformer and coefficient head
│   ├── basis.py                 # 2D velocity basis and spectral mix-norm
│   ├── basis3d.py               # 3D velocity basis and spectral mix-norm
│   ├── dataset.py               # four 2D partition families and samplers
│   ├── dataset3d.py             # four 3D partition families and samplers
│   ├── util.py                  # logging and shared evaluation lattice
│   └── eulerian/
│       ├── __init__.py
│       ├── pde_solver.py        # 2D finite-volume solver and control optimizer
│       └── pde_solver3d.py      # 3D finite-volume solver
└── example/
    ├── __init__.py
    ├── test.py                   # one 2D case, controls, and velocity plots
    ├── test_multi.py             # multiple 2D train/test cases; shared helpers
    ├── test_cross_all_2d.py      # four-family 2D figure
    ├── test_cross_family.py      # selected 2D family-pair comparison
    ├── test_fourier_mixed.py     # additional 2D partition examples
    ├── test3d.py                 # 3D particle evaluation
    ├── test3d_density.py         # 3D Eulerian evaluation; shared helpers
    ├── test3d_cross_family.py    # four-family 3D Eulerian figure
    ├── tests3d_sanity.py         # basis and mix-norm checks
    ├── make_videos_2d.py         # optional particle / Eulerian animations
    ├── compare_particle_pde.py  # verification under the learned control
    └── compare_pde_particle.py  # direct PDE optimization comparison; shared helpers
```

Use `python train.py --help`, `python train3d.py --help`, or `python -m example.<script_name> --help` for options. Examples also support direct invocation, such as `python example/test.py --help`. The 2D PDE optimizer has its own interface: `python -m model.eulerian.pde_solver --help`. Files named `test*.py` are evaluation tools, apart from the numerical checks in `example/tests3d_sanity.py`.

The optional Slurm launchers contain site-specific module, Conda environment, partition, and account settings; adapt them before submission. Submit from `Code-mixing-OP/`, or set `CODE_DIR` to its absolute path. Set `LAM_KIN` explicitly to the manuscript value:

```bash
FAMILY=rotation_line LAM_KIN=2e-3 sbatch run_train_family.sh
FAMILY=rotation_plane LAM_KIN=2e-2 sbatch run_train_3d.sh
```

## Citation

If you use this code in your research, please cite

```bibtex
@article{hu2026self,
  title={Self-Supervised In-Context Operator Learning for Optimal Mixing in Incompressible Flows},
  author={Hu, Weiwei and Lai, Rongjie and Li, Ziqian},
  journal={arXiv preprint arXiv:2610.25556},
  year={2026}
}
```

## License

Released for academic use accompanying the manuscript above.
