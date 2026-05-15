# CP-Dynamics: Contact-Point Dynamics for Embodiment-Invariant VLA Models

> **Thesis:** Environment dynamics are robot-invariant. The physics of "what happens to an object when it is contacted" is determined by the contact interaction itself — not by which robot produced it. A VLA that learns dynamics in this contact-centric, embodiment-invariant space will transfer across morphologies without per-robot fine-tuning.

---

## 1. The Problem

Current VLAs (Octo, OpenVLA, π₀) output **joint-space commands** specific to each robot. When transferring to a new robot:

- Different DOFs (6 vs 7 vs 16)
- Different kinematics (spherical wrist vs offset elbow)
- Different workspaces (near-body vs extended)

The result: **retraining the action head** for every new robot. The perception transfers; the control doesn't.

**The bottleneck:** VLAs learn policies (observation → action) in embodiment-specific action spaces. Nobody learns dynamics (state + action → next state) in an embodiment-invariant space.

---

## 2. The Insight

When **any** robot manipulates an object, the object's motion is governed by **three things only**:

1. **Where** on the object surface contact occurs
2. **What wrench** (force/torque) is applied at each contact
3. **The object's own state** (pose, shape, mass, friction)

The robot's joint count, link lengths, actuator types, and kinematic structure are **irrelevant** to the object dynamics once contact is established.

**The robot is merely a wrench delivery mechanism.**

---

## 3. Why This Is Novel

The cross-embodiment VLA landscape in 2025–2026 has four approaches. None achieves all three properties simultaneously:

| Approach | Example | Shares Perception | Shares Dynamics | Cross-Morphology |
|:---|:---|:---:|:---:|:---:|
| Co-training + per-robot heads | RT-X (2023), Octo (2024), π₀ (2024) | ✅ | ❌ | Partial |
| Universal action spaces | UniAct (CVPR 2025), OPFA (ICRA 2026) | ✅ | ❌ | ✅ |
| Cross-embodiment world models | He et al. (Nov 2025) | ✅ | ✅ | ❌ (hands only) |
| **CP-Dynamics (proposed)** | — | ✅ | **✅** | **✅** |

### Detailed comparison with closest work

**He et al. — "Scaling Cross-Embodiment World Models" (Nov 2025, Hao Su group, UCSD)**
- Conjecture: "environment dynamics are embodiment-invariant"
- Represent embodiments as 3D particles, actions as particle displacements
- Graph-based world model across dexterous hands
- **Gap:** Limited to hand manipulation. Particle abstraction doesn't generalize to arms, mobile manipulators. We use contact-point state instead.

**UniAct — "Universal Actions for Enhanced Embodied Foundation Models" (CVPR 2025)**
- Defines universal action space capturing shared structural features
- Per-embodiment decode heads translate to robot-specific commands
- 0.5B model outperforms 14× larger SOTA
- **Gap:** Shared action space, not shared dynamics. The model still learns per-robot dynamics implicitly.

**OPFA — "One-Policy-Fits-All" (ICRA 2026)**
- Geometry-aware latent representation (GaLR) via 3D convolutions + transformers
- Unified latent retargeting decoder — NO per-embodiment decoder tuning
- Tested across 11 end-effectors, 50%+ improvement from cross-embodiment co-training
- **Gap:** Latent actions, not latent dynamics. The model doesn't predict what happens to the world.

**UniVLA — "Learning to Act Anywhere with Task-centric Latent Actions" (RSS 2025)**
- Derives latent actions from videos using DINO feature space
- SOTA with 1/20th of OpenVLA's pretraining compute
- **Gap:** Latent actions from video, not dynamics. No state-transition prediction.

**AnyCar to Anywhere (UC Berkeley, Sep 2024)**
- Transformer-based dynamics model for wheeled vehicles
- Cross-platform transfer within wheeled robots
- **Gap:** Only wheeled vehicles. Doesn't transfer to manipulation.

### The gap (confirmed by exhaustive search)

**Nobody has learned a dynamics model that:**
1. Is embodiment-invariant (same model works for any robot)
2. Works across fundamentally different morphologies (not just different hands)
3. Predicts state transitions (not just actions)

The intersection of these three properties is **empty**.

---

## 4. The Mechanism: Contact-Point Dynamics (CPD)

### 4.1 Representation: Contact-Point State

```
C ∈ ℝ^(N×9)    N = max 8 contact points

Each row encodes one contact:
  p_c ∈ ℝ³     position on object surface (object frame)
  f_c ∈ ℝ³     applied force vector (object frame)
  n_c ∈ ℝ³     contact normal + slip indicator
```

**Why contact-point state, not task-space pose:**
- Task-space (end-effector pose) breaks for multi-contact scenarios (dexterous hands)
- Task-space assumes similar end-effectors — not true for arm vs. hand vs. mobile manipulator
- Contact-point state is the **minimal sufficient statistic** for embodiment-invariant dynamics

**Why contact-point state, not particles (He et al.):**
- Parters require explicit 3D geometry of the robot — hard to extract from real robots
- Contact-point state can be estimated from joint torques via Jacobian: `τ = J^T F`
- Contact-point state works for ANY robot with force sensing (or joint torque estimation)

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Shared Dynamics Model                        │
│                 (Transformer, ~1M params)                    │
│                                                             │
│  Input: tokens = [z_obj; C₁; C₂; ...; C₈]                 │
│         z_obj: object state embedding (128-dim)              │
│         C_i: contact-point state (9-dim each)               │
│                                                             │
│  TransformerEncoder (6 layers, 8 heads, d=256)              │
│                                                             │
│  Output: Δz_obj = MLP(object_token)     ← object state Δ   │
│          C' = MLP(contact_tokens)       ← contact evolution │
│          wrench = MLP(object_token)      ← net force (F=ma) │
│                                                             │
│  0 embodiment-specific parameters                           │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌───────────────┐  ┌──────────────────┐
│  Franka Panda   │  │  Allegro Hand │  │  Stretch RE2     │
│  (7-DOF arm)    │  │  (16-DOF)     │  │  (mobile manip)  │
│                 │  │               │  │                  │
│  Contact        │  │  Contact      │  │  Contact         │
│  Estimator      │  │  Estimator    │  │  Estimator       │
│  (~5K params)   │  │  (~5K params) │  │  (~5K params)    │
└─────────────────┘  └───────────────┘  └──────────────────┘

Per-robot contact estimator:
  Input:  (q, q̇, τ) — joint state, velocities, torques
  Output: C ∈ ℝ^(N×9) — estimated contact-point state
```

### 4.3 Training Pipeline

**Stage 1: Contact Estimator (per-robot, supervised in simulation)**

```
E_e: (q, q̇, τ) → Ĉ ∈ ℝ^(N×9)
```

- Small MLP (2 layers, 256 hidden)
- Supervised with ground-truth contact data from physics engine
- ~1 hour of training per robot

**Stage 2: Shared Dynamics Model (trained on multi-robot data)**

```
L = λ₁ · MSE(Δz_pred, Δz_gt)      # dynamics loss
  + λ₂ · MSE(C_pred, C_{t+1})      # contact evolution loss
  + λ₃ · L_physics(wrench, Δz)     # F=ma consistency
```

- L_physics: enforce Newton's second law in latent space
- Trained on pooled data from all robots (batched, balanced)
- ~10 hours on single GPU

**Stage 3: VLA Integration (contact target head)**

- New MLP head on VLA: visual features → target contact state C_target
- CPD model simulates forward: given current contacts and target, predict outcome
- If prediction doesn't match goal, refine C_target via gradient descent through differentiable CPD
- Solve IK from refined C_target to get final action

### 4.4 Transfer to New Robot

```
1. Freeze dynamics model (zero-shot)
2. Train new robot's contact estimator (~1 hour, simulation)
3. Run
```

Three options for contact estimation on new robots:
- **Simulation available:** Supervised training with ground-truth contacts
- **Real robot, no tactile:** Fine-tune from closest existing estimator using contact consistency loss
- **Fully zero-shot:** Forward kinematics + Jacobian transpose: `τ = J^T F`

---

## 5. Measurable Hypotheses

**H1 (Dynamics Invariance):** A dynamics model trained jointly on Franka Panda (7-DOF arm) and Allegro Hand (16-DOF dexterous hand) data achieves ≤10% object state prediction error degradation when evaluated on tasks executed by either robot, compared to per-robot dynamics baselines.

**H2 (Transfer Efficiency):** A new robot (UR5e, 6-DOF) achieves ≥80% of its single-robot baseline success rate after training ONLY its contact estimator on 50 demonstrations, while the dynamics model and VLA remain frozen.

**H3 (Zero-Shot Object Generalization):** The dynamics model, trained on pick-and-place with cubes and cylinders, achieves ≥60% success rate on a novel object (sphere) when used with any of the three robots.

---

## 6. The Minimal Experiment

### 6.1 Core Claim (falsifiable)

Object dynamics during pushing are a function of (object state, task-space interaction) — NOT of which robot arm is doing the pushing. If true, a single dynamics model should predict object motion from both robots with zero robot-specific adaptation.

### 6.2 Setup

| Component | Choice | Why |
|:---|:---|:---|
| **Robots** | Franka Panda (7-DOF) + UR5e (6-DOF) | Different DOFs, different kinematics, same gripper type |
| **Environment** | ManiSkill3 (SAPIEN) | 20+ robots, same-task-swap via `gym.make(robot=...)`, 30K+ FPS, ground-truth contacts |
| **Task** | PushCube-v1 | Simple dynamics, clear success metric |
| **Data** | 2000 trajectories/robot, scripted policy | ~30 min wall-clock |
| **Compute** | ~4 hours on 1× A100 | Accessible |

**Why ManiSkill3:**

| Environment | Same Task × N Robots | GT State | Force/Torque | Speed |
|:---|:---:|:---:|:---:|:---:|
| **ManiSkill3** ✅ | ✅ (built-in swap) | ✅ | ✅ (SAPIEN contacts) | 30K+ FPS |
| Isaac Lab | ✅ (configurable) | ✅ | ✅ (native) | GPU parallel |
| RoboTwin 2.0 | ✅ (5 platforms) | ✅ | ⚠️ | GPU parallel |
| MetaWorld ❌ | ❌ (Sawyer only) | ✅ | ⚠️ | CPU |
| LIBERO ❌ | ❌ (Franka only) | ✅ | ⚠️ | CPU |

### 6.3 Data Collection

Generate N=2000 trajectories per robot using a scripted pushing policy:
- Randomize: cube position, push direction, push speed
- Record per timestep: `(object_pose_7D, ee_pose_7D, ee_delta_3D, gripper_state)`
- ~50 timesteps per trajectory → 100k transitions per robot

### 6.4 Architecture (ALL shared, zero robot-specific parameters)

```
object_pose ────▶ Obs Encoder ──▶ z_t ∈ ℝ³² (latent object state)
ee_pose ────────▶ (shared)
gripper ────────▶

ee_delta_3D ────▶ Action Encoder ──▶ a ∈ ℝ¹² (latent action)
(task-space)      (shared)

(z_t, a) ───────▶ Dynamics Model ──▶ z_{t+1}
                   (shared)

z_{t+1} ────────▶ Obs Decoder ──▶ object_pose_{t+1}
                   (shared)
```

**Key:** Action = end-effector delta in task space (Δx, Δy, Δz), NOT joint-space commands. The same Δee produces the same object motion regardless of which arm generated it.

### 6.5 The Three Models (controlled comparison)

| Model | Training Data | What it tests |
|:---|:---|:---|
| M_Franka | Franka only (2000 traj) | Upper bound on Franka |
| M_UR5e | UR5e only (2000 traj) | Upper bound on UR5e |
| **M_mixed** | **Both robots (4000 traj, shuffled)** | **The hypothesis: shared dynamics works** |

All three: identical architecture, lr, epochs. Adam, lr=1e-3, 100 epochs, batch 256. ~2 hours on A100.

### 6.6 Evaluation

**Transfer Ratio (the money metric)**
```
Transfer Ratio = MSE(mixed→Franka) / MSE(Franka→Franka)
```

**Wrong-robot baseline (the floor)**
```
M_UR5e applied to Franka test data → should FAIL
```

**Latent dynamics alignment**
```
For matched scenarios (same cube pose, same push direction):
cosine_sim(Δz_Franka, Δz_UR5e) — should be > 0.7
```

### 6.7 Pre-Registered Success Criteria

| Criterion | Threshold | Meaning |
|:---|:---|:---|
| Transfer Ratio | ≤ 1.15 | Mixed model ≤15% worse than single-robot |
| Wrong-robot fails | MSE > 2× baseline | Robot-specific models DON'T transfer |
| Latent alignment | cosine > 0.7 | Shared latent space captures same dynamics |

**Pass:** ALL three met → full-scale VLA training justified.
**Fail:** ANY fails → redirect to robot-specific representations.

### 6.8 Compute Budget

| Step | Time | Hardware |
|:---|:---|:---|
| Data generation | ~30 min | GPU (ManiSkill3 parallel envs) |
| Training (3 models) | ~2 hours | Single A100 |
| Evaluation + analysis | ~1 hour | CPU/GPU |
| **Total** | **~4 hours** | **1× A100** |

---

## 7. Honest Assessment

### What's genuinely novel
The explicit decomposition of dynamics (shared, contact-centric) from kinematics (per-robot, contact estimation) in a VLA context. He et al. proved the concept for hands; we extend it to arbitrary morphologies.

### What's incremental
"Physics is robot-invariant" is obvious. The contribution is proving it works for VLA transfer, not inventing the concept.

### What I'm uncertain about
1. **Contact estimation accuracy:** Is joint-torque-based contact estimation accurate enough? Real torque sensors have ~5% error. Mitigation: the transformer's attention can learn to be robust to noisy contacts.
2. **Information preservation:** Does the latent action space preserve enough information for precise manipulation? Mitigation: multi-head decoder (pose delta, particle displacement, point cloud delta).
3. **Environment comparability:** Does ManiSkill3 actually support the same task across different robots with comparable difficulty? Mitigation: verify via scripted policy performance before dynamics training.

### What would make this foundational
If the dynamics model predicts object motion for a robot it has NEVER seen, with only a new contact estimator trained on a handful of demos — that's not "5% better." That's "physics transfers across morphologies."

---

## 8. Related Work (Exhaustive)

### Cross-Embodiment VLA Models

| Paper | Date | Venue | Key Contribution | Shares Dynamics? |
|:---|:---|:---|:---|:---:|
| RT-X / Open X-Embodiment | Oct 2023 | ICRA 2024 | Co-training on 22 robots, shared perception | ❌ |
| Octo | May 2024 | arXiv | Generalist policy, modular action heads | ❌ |
| OpenVLA | Jun 2024 | arXiv | Open-source 7B VLA, discretized actions | ❌ |
| π₀ | Oct 2024 | RSS 2025 | Flow-matching VLA, multi-robot | ❌ |
| X-VLA | Oct 2025 | ICLR 2026 | Soft-prompted cross-embodiment | ❌ |
| UniAct | Jan 2025 | CVPR 2025 | Universal action space | ❌ |
| OPFA | Mar 2026 | ICRA 2026 | Geometry-aware latent actions, 11 end-effectors | ❌ |
| UniVLA | May 2025 | RSS 2025 | Task-centric latent actions from video | ❌ |
| FAST/π₀-FAST | Jan 2025 | RSS 2025 | Universal action tokenizer (DCT) | ❌ |
| π₀.₇ | Apr 2026 | arXiv | Steerable generalist VLA | ❌ |
| GR00T N1 | Mar 2025 | arXiv | NVIDIA humanoid foundation model | ❌ |

### Cross-Embodiment Dynamics & World Models

| Paper | Date | Venue | Key Contribution | Scope |
|:---|:---|:---|:---|:---|
| **He et al.** | Nov 2025 | arXiv | Particle-based embodiment-invariant world model | Hands only |
| AnyCar | Sep 2024 | arXiv | Dynamics model for wheeled vehicles | Wheeled only |
| Lagrangian GNN | NeurIPS 2022 | NeurIPS | Energy-conserving dynamics for articulated bodies | Simple bodies |
| TraceGen | Nov 2025 | arXiv | World model in 3D trace-space | Video-based |
| LAC-WM | Dec 2025 | OpenReview | Latent action world model | Video-based |
| Motus | Dec 2025 | arXiv | Unified latent action world model | Video-based |
| RoboPack | RSS 2024 | RSS | Tactile-informed dynamics (single robot) | Single robot |

### Cross-Embodiment Policies

| Paper | Date | Venue | Key Contribution |
|:---|:---|:---|:---|
| Ai et al. — Embodiment Scaling Laws | May 2025 | CoRL 2025 | Policy scaling across 1000 locomotion morphologies |
| Yang, Finn — Data Analogies | Mar 2026 | CoRL 2026 | Paired demos matter for morphology transfer |
| Canonical Rep. (Wei et al.) | Feb 2026 | arXiv | Cross-hand grasping policy via canonical URDF |
| Equivariant Flow (Siebenborn) | May 2026 | arXiv | Symmetry-equivariant policy for bimanual |
| XHugWBC | Feb 2026 | arXiv | Cross-humanoid whole-body control |
| GCNT | May 2025 | arXiv | Graph-based morphology-agnostic policy (locomotion) |

---

## 9. Impact

If this works:
- **New robot deployment:** Train contact estimator (~1 hour). Dynamics model transfers frozen.
- **Scaling:** Adding a new robot doesn't require retraining the dynamics model. Just add a new contact estimator.
- **Foundation model for dynamics:** The dynamics model becomes a reusable component, like a pretrained VLM is for perception.

If this fails:
- We learn exactly where embodiment invariance breaks — still a valuable negative result for the field.
