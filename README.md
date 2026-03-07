# SoftGear

**Soft Gears, Hard Results** — Progressive depth training with gradual hardening.

Inspired by the [differential analyzer](https://en.wikipedia.org/wiki/Differential_analyser), SoftGear trains deep transformers by mounting gears one at a time, gradually hardening older ones so each gear develops a distinct role.

## How It Works

A **gear** is a group of Transformer layers. Training proceeds in phases:

```
Phase 1: Mount G1 (1 layer, soft)    — learns coarse structure
Phase 2: Mount G2 (2 layers, soft)   — G1 hardens, G2 refines
Phase 3: Mount G3 (3 layers, soft)   — G1/G2 harden further, G3 adds detail
```

Two mechanisms work together:

- **Identity init** — New gears start as identity functions (zero output projections), preserving the existing chain's behavior.
- **Gradual hardening** — Older gears get exponentially decayed learning rates (`lr *= decay` per phase), protecting their learned representations.

```
Phase K learning rates:
  G1: lr * decay^(K-1)   (hardest — locked in)
  G2: lr * decay^(K-2)
  ...
  GK: lr * 1.0           (softest — actively learning)
```

## Project Structure

```
src/softgear/
├── models/
│   ├── analyzer.py       # Analyzer = encoder + chain + decoder
│   ├── chain.py           # Chain = sequential stack of gears
│   └── gear.py            # Gear = N Pre-LN Transformer layers
├── training/
│   ├── progressive_depth.py  # ProgressiveDepthScheduler (hardening logic)
│   └── trainer.py             # Training loop
├── tasks/
│   └── sudoku/            # Sudoku 4x4 / 9x9 task
├── config.py              # Dataclass configs
└── cli.py                 # CLI entry point

notebooks/                 # Colab experiment notebooks
├── exp1-hardening.ipynb   # Compare 5 hardening strategies
├── exp2-ablation.ipynb    # Ablation: identity init, decay rate, gear count
├── exp3-curriculum.ipynb  # Curriculum vs hardening
└── exp4-scale.ipynb       # Model size generalization
```

## Metaphor-to-Code Mapping

| Gear Metaphor              | Code                                   |
| -------------------------- | -------------------------------------- |
| Gear                       | `Gear` — Pre-LN Transformer layer(s)   |
| Gear chain                 | `Chain` — `nn.ModuleList` of Gears     |
| Differential analyzer      | `Analyzer` — encoder + chain + decoder |
| Mount a gear               | `Chain.mount(gear)`                    |
| Hardness                   | `param_group["lr"]`                    |
| Gradual hardening          | `ProgressiveDepthScheduler`            |
| Blank gear (identity init) | `Gear(identity_init=True)`             |

## Quick Start

```bash
# Install
uv pip install git+https://github.com/byExist/softgear.git

# Download dataset
softgear download sudoku9

# Train with gradual hardening (default)
softgear train --task sudoku9 --hardening gradual

# Train without protection (baseline)
softgear train --task sudoku9 --hardening none

# Evaluate
softgear eval checkpoints/best.pt
```

### Hardening Strategies

```bash
softgear train --task sudoku9 --hardening gradual       # per-layer lr decay (default)
softgear train --task sudoku9 --hardening none           # all layers same lr
softgear train --task sudoku9 --hardening freeze         # existing layers lr=0
softgear train --task sudoku9 --hardening binary         # old 0.4x / new 1.0x
softgear train --task sudoku9 --hardening from_scratch   # no progressive stacking
```

### Key Options

```bash
--num-gears 7          # number of progressive phases
--lr-decay 0.5         # hardening decay factor
--scale 1.5            # resolution scale: gear i gets round(scale^i) layers
--identity-init        # identity init for new gears (default: on)
--no-identity-init     # random init (ablation)
--curriculum           # sort training data by difficulty
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.10
- GPU recommended (Colab T4 works)
