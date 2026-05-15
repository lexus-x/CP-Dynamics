<p align="center">
  <img src="assets/banner.svg" width="100%" alt="CP-Dynamics: Contact-Point Dynamics for Embodiment-Invariant VLA Models"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-2026-blue" alt="arXiv"/></a>
  <a href="#"><img src="https://img.shields.io/badge/venue-ICML%202026-purple" alt="ICML 2026"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.10+-yellow" alt="Python 3.10+"/></a>
  <a href="#"><img src="https://img.shields.io/badge/simulation-ManiSkill3-orange" alt="ManiSkill3"/></a>
</p>

---

## TL;DR

**Environment dynamics are robot-invariant.** The physics of pushing a cup is the same whether you use a Franka, a UR5, or a dexterous hand. We learn a dynamics model in **contact-point space** — an embodiment-invariant representation — that transfers across morphologies without per-robot fine-tuning. A new robot only needs a tiny contact estimator (~5K params, 1 hour training). The dynamics model transfers frozen.

---

## The Problem

Current VLAs (Octo, OpenVLA, π₀) output **joint-space commands** specific to each robot. Transferring to a new robot means retraining the action head. The perception transfers; the control doesn't.

**Why?** Because VLAs learn policies (observation → action) in embodiment-specific action spaces. Nobody learns **dynamics** (state + action → next state) in an embodiment-invariant space.

---

## The Insight

When **any** robot manipulates an object, the object's motion is governed by **three things only**:

1. **Where** on the object surface contact occurs
2. **What wrench** (force/torque) is applied at each contact
3. **The object's own state** (pose, shape, mass, friction)

The robot's joint count, link lengths, and kinematic structure are **irrelevant** to the object dynamics once contact is established.

> **The robot is merely a wrench delivery mechanism.**

<p align="center">
  <img src="assets/architecture.svg" width="100%" alt="CP-Dynamics Architecture"/>
</p>

---

## Why This Is Novel

<p align="center">
  <img src="assets/venn.svg" width="100%" alt="Novelty: Intersection of Three Properties"/>
</p>

| Approach | Shares Perception | Shares Dynamics | Cross-Morphology |
|:---|:---:|:---:|:---:|
| Co-training (RT-X, Octo, π₀) | ✅ | ❌ | Partial |
| Universal actions (UniAct, OPFA) | ✅ | ❌ | ✅ |
| Cross-embodiment world models (He et al.) | ✅ | ✅ | ❌ (hands only) |
| **CP-Dynamics (ours)** | ✅ | **✅** | **✅** |

**Nobody has demonstrated all three simultaneously.** The intersection is empty. We fill it.

### Closest work and why it's not enough

- **He et al. (Nov 2025):** Particle-based dynamics for dexterous hands. Conjectures "dynamics are embodiment-invariant" but only proves it for hands. We use contact-point state instead of particles — works for any morphology.
- **UniAct (CVPR 2025):** Universal action space with per-embodiment decoders. Shared action, not shared dynamics.
- **OPFA (ICRA 2026):** Geometry-aware latent actions across 11 end-effectors. Unified decoder. Still policy-level, not dynamics-level.
- **UniVLA (RSS 2025):** Latent actions from video. No state-transition prediction.

---

## The Mechanism

### Contact-Point State (embodiment-invariant representation)

```
C ∈ ℝ^(N×9)    N = max 8 contact points

Each row:
  p_c ∈ ℝ³     contact position on object surface (object frame)
  f_c ∈ ℝ³     applied force vector (object frame)
  n_c ∈ ℝ³     contact normal + slip indicator
```

A 7-DOF arm making one contact and a 16-DOF hand making three contacts both populate different rows of the **same matrix**. The dynamics model doesn't know or care how the contacts were achieved.

### Shared Dynamics Model

```
tokens = [z_obj; C₁; C₂; ...; C₈]    ← object + contact tokens
         ↓
         Transformer Encoder (6 layers, 8 heads, d=256)
         ↓
Δz_obj = MLP(object_token)            ← predicted object state change
C' = MLP(contact_tokens)              ← predicted contact evolution
wrench = MLP(object_token)            ← net wrench (F=ma consistency check)
```

**0 embodiment-specific parameters.** ~1M params total.

### Per-Robot Contact Estimator

```
Input:  (q, q̇, τ) — joint state, velocities, torques
         ↓
         MLP (2 layers, 256 hidden, ~5K params)
         ↓
Output:  C ∈ ℝ^(N×9) — estimated contact-point state
```

The **only** component that is per-embodiment. Trained in simulation with ground-truth contacts. ~1 hour per robot.

### Transfer to New Robot

```
1. Freeze dynamics model (zero-shot)
2. Train new robot's contact estimator (~1 hour)
3. Run
```

---

## The Experiment

### Setup

| Component | Choice | Why |
|:---|:---|:---|
| **Robots** | Franka Panda (7-DOF) + UR5e (6-DOF) | Different DOFs, different kinematics |
| **Environment** | ManiSkill3 | 20+ robots, same-task-swap, 30K+ FPS, ground-truth contacts |
| **Task** | PushCube-v1 | Simple dynamics, clear success metric |
| **Data** | 2000 trajectories/robot | ~30 min wall-clock |
| **Compute** | ~4 hours on 1× A100 | Accessible |

### Controlled Comparison

| Model | Training Data | What it tests |
|:---|:---|:---|
| M_Franka | Franka only | Upper bound |
| M_UR5e | UR5e only | Upper bound |
| **M_mixed** | **Both (shuffled)** | **The hypothesis** |

### Pre-Registered Success Criteria

| Criterion | Threshold | Meaning |
|:---|:---|:---|
| **Transfer Ratio** | ≤ 1.15 | Mixed model ≤15% worse than single-robot |
| **Wrong-robot fails** | MSE > 2× baseline | Robot-specific models DON'T transfer |
| **Latent alignment** | cosine > 0.7 | Shared latent space captures same dynamics |

**Pass:** All three → full-scale VLA training justified.
**Fail:** Any → redirect to robot-specific representations.

---

## Quick Start

```bash
git clone https://github.com/lexus-x/CP-Dynamics.git
cd CP-Dynamics
pip install -e .

# 1. Generate data (~30 min)
python scripts/generate_data.py --robot panda --episodes 2000
python scripts/generate_data.py --robot ur5e --episodes 2000

# 2. Train contact estimators (~1 hour)
python scripts/train_contact_estimator.py --robot panda
python scripts/train_contact_estimator.py --robot ur5e

# 3. Train dynamics models (~2 hours)
python scripts/train_dynamics.py --data mixed --epochs 100
python scripts/train_dynamics.py --data franka_only --epochs 100
python scripts/train_dynamics.py --data ur5e_only --epochs 100

# 4. Evaluate
python scripts/evaluate.py --model mixed --robot franka
python scripts/evaluate.py --model mixed --robot ur5e
python scripts/evaluate.py --model franka_only --robot ur5e  # wrong-robot baseline
```

---

## Project Structure

```
CP-Dynamics/
├── assets/              # SVG visuals (banner, architecture, venn)
├── docs/
│   ├── PROPOSAL.md      # Full research proposal (30+ papers)
│   ├── LITERATURE.md    # Exhaustive literature review
│   ├── EXPERIMENT.md    # Detailed experiment protocol
│   └── THEORY.md        # Theoretical motivation
├── src/dynamics/
│   ├── model.py         # Shared dynamics model (Transformer)
│   └── contact_encoder.py  # Per-robot contact estimators
├── scripts/             # Training and evaluation scripts
├── README.md
└── LICENSE (MIT)
```

---

## Related Work

<details>
<summary><b>Cross-Embodiment VLA Models (11 papers)</b></summary>

| Paper | Date | Venue | Key Contribution |
|:---|:---|:---|:---|
| RT-X / Open X-Embodiment | Oct 2023 | ICRA 2024 | Co-training on 22 robots |
| Octo | May 2024 | arXiv | Generalist policy, modular action heads |
| OpenVLA | Jun 2024 | arXiv | Open-source 7B VLA |
| π₀ | Oct 2024 | RSS 2025 | Flow-matching VLA, multi-robot |
| X-VLA | Oct 2025 | ICLR 2026 | Soft-prompted cross-embodiment |
| UniAct | Jan 2025 | CVPR 2025 | Universal action space |
| OPFA | Mar 2026 | ICRA 2026 | Geometry-aware latent actions, 11 end-effectors |
| UniVLA | May 2025 | RSS 2025 | Latent actions from video |
| FAST/π₀-FAST | Jan 2025 | RSS 2025 | Universal action tokenizer |
| π₀.₇ | Apr 2026 | arXiv | Steerable generalist VLA |
| GR00T N1 | Mar 2025 | arXiv | NVIDIA humanoid foundation model |
</details>

<details>
<summary><b>Cross-Embodiment Dynamics & World Models (7 papers)</b></summary>

| Paper | Date | Venue | Key Contribution | Scope |
|:---|:---|:---|:---|:---|
| He et al. | Nov 2025 | arXiv | Particle-based embodiment-invariant world model | Hands only |
| AnyCar | Sep 2024 | arXiv | Dynamics model for wheeled vehicles | Wheeled only |
| Lagrangian GNN | NeurIPS 2022 | NeurIPS | Energy-conserving dynamics | Simple bodies |
| TraceGen | Nov 2025 | arXiv | World model in 3D trace-space | Video-based |
| LAC-WM | Dec 2025 | OpenReview | Latent action world model | Video-based |
| Motus | Dec 2025 | arXiv | Unified latent action world model | Video-based |
| RoboPack | RSS 2024 | RSS | Tactile-informed dynamics | Single robot |
</details>

<details>
<summary><b>Cross-Embodiment Policies (6 papers)</b></summary>

| Paper | Date | Venue | Key Contribution |
|:---|:---|:---|:---|
| Ai et al. — Scaling Laws | May 2025 | CoRL 2025 | Policy scaling across 1000 morphologies |
| Yang, Finn — Data Analogies | Mar 2026 | CoRL 2026 | Paired demos matter for transfer |
| Canonical Rep. (Wei et al.) | Feb 2026 | arXiv | Cross-hand grasping policy |
| Equivariant Flow | May 2026 | arXiv | Symmetry-equivariant bimanual policy |
| XHugWBC | Feb 2026 | arXiv | Cross-humanoid whole-body control |
| GCNT | May 2025 | arXiv | Graph-based morphology-agnostic policy |
</details>

---

## Impact

**If this works:**
- New robot deployment = 1 hour of contact estimator training
- Dynamics model becomes a reusable foundation component
- Scaling: adding robots doesn't require retraining dynamics

**If this fails:**
- We learn exactly where embodiment invariance breaks — still a valuable negative result

---

## Citation

```bibtex
@article{cpdynamics2026,
  title={CP-Dynamics: Contact-Point Dynamics for Embodiment-Invariant Vision-Language-Action Models},
  author={lexus-x},
  year={2026},
  url={https://github.com/lexus-x/CP-Dynamics}
}
```

---

## License

[MIT](LICENSE)
