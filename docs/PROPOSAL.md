# CP-Dynamics: Research Proposal

> **Thesis:** Environment dynamics are robot-invariant. The physics of "what happens to an object when it is contacted" is determined by the contact interaction — not by which robot produced it.

---

## 1. Problem

Current VLAs learn policies in embodiment-specific action spaces. Transferring to a new robot requires retraining. The perception transfers; the control doesn't.

**Root cause:** VLAs learn observation → action mappings in joint space. Joint space is embodiment-specific (6-DOF vs 7-DOF vs 16-DOF). No shared dynamics model exists.

---

## 2. Insight

Object dynamics during manipulation are determined by:
1. Where contact occurs on the object surface
2. What wrench is applied at each contact
3. The object's own state (pose, shape, mass, friction)

The robot's kinematic structure is irrelevant once contact is established. **The robot is a wrench delivery mechanism.**

---

## 3. Gap Analysis

### What exists

| Paper | Type | Cross-morphology? | Shared dynamics? | Limitation |
|:---|:---|:---:|:---:|:---|
| He et al. (Nov 2025) | Dynamics model | ❌ (hands only) | ✅ | Particle representation, hands only |
| UniAct (CVPR 2025) | Universal actions | ✅ | ❌ | Shared action space, not dynamics |
| OPFA (ICRA 2026) | Latent actions | ✅ | ❌ | Unified decoder, still policy-level |
| UniVLA (RSS 2025) | Latent actions | ✅ | ❌ | From video, no state prediction |
| RT-X (ICRA 2024) | Co-training | Partial | ❌ | Shared perception only |
| AnyCar (Sep 2024) | Dynamics model | ❌ | ✅ | Wheeled vehicles only |

### What's missing

**The intersection of "dynamics model + embodiment-invariant + arbitrary morphologies" is empty.**

Nobody has learned a dynamics model that:
- Is embodiment-invariant (same model for any robot)
- Works across fundamentally different morphologies
- Predicts state transitions (not just actions)

---

## 4. Mechanism: Contact-Point Dynamics

### 4.1 Representation

```
Contact-Point State: C ∈ ℝ^(N×9), N = max 8

Each row: [position(3), force(3), normal+slip(3)]

Object State: z_obj ∈ ℝ^128 (learned from vision)
```

**Why contact-point state:**
- Embodiment-invariant: different robots populate different rows of the same matrix
- Minimal sufficient statistic for object dynamics
- Estimable from joint torques: τ = J^T F
- Unlike task-space pose: works for multi-contact (hands)
- Unlike particles (He et al.): doesn't require explicit 3D robot geometry

### 4.2 Architecture

**Shared dynamics model (~1M params, 0 embodiment-specific):**
```
Input:  tokens = [z_obj; C₁; C₂; ...; C₈]
        ↓
        Transformer Encoder (6 layers, 8 heads, d=256)
        ↓
Output: Δz_obj (object state change)
        C' (next contact states)
        wrench (net force, for F=ma check)
```

**Per-robot contact estimator (~5K params):**
```
Input:  (q, q̇, τ) — joint state, velocities, torques
        ↓
        MLP (2 layers, 256 hidden)
        ↓
Output: C ∈ ℝ^(N×9) — contact-point state
```

### 4.3 Training

**Stage 1 — Contact estimator (per-robot, supervised)**
- Ground-truth contacts from physics engine
- ~1 hour per robot

**Stage 2 — Dynamics model (shared, multi-robot)**
- Loss = MSE(Δz) + MSE(C') + λ·L_physics(F=ma)
- Pooled data from all robots, balanced batches
- ~10 hours on single GPU

**Stage 3 — VLA integration**
- Contact target head: visual features → C_target
- CPD refines C_target → action via differentiable forward pass

### 4.4 Transfer

```
1. Freeze dynamics model (zero-shot)
2. Train new robot's contact estimator (~1 hour)
3. Run
```

---

## 5. Hypotheses

**H1 (Dynamics invariance):** Mixed model (Franka + UR5e) achieves ≤10% prediction error degradation vs. per-robot models.

**H2 (Transfer efficiency):** New robot (UR5e) achieves ≥80% baseline with only 50 demos for its contact estimator. Dynamics frozen.

**H3 (Object generalization):** Dynamics model generalizes to novel objects (not seen during training) with ≥60% success.

---

## 6. Experiment

### Setup

| Component | Choice |
|:---|:---|
| Robots | Franka Panda (7-DOF) + UR5e (6-DOF) |
| Environment | ManiSkill3 (SAPIEN, 30K+ FPS, ground-truth contacts) |
| Task | PushCube-v1 |
| Data | 2000 trajectories/robot, scripted policy |
| Compute | ~4 hours on 1× A100 |

### Protocol

1. Collect data from both robots
2. Train three models: M_Franka, M_UR5e, M_mixed
3. Evaluate Transfer Ratio, wrong-robot baseline, latent alignment

### Success Criteria (pre-registered)

| Criterion | Threshold |
|:---|:---|
| Transfer Ratio | ≤ 1.15 |
| Wrong-robot fails | MSE > 2× baseline |
| Latent alignment | cosine > 0.7 |

---

## 7. Honest Assessment

**Novel:** The explicit decomposition of shared dynamics (contact-centric) from per-robot kinematics in a VLA context.

**Incremental:** "Physics is robot-invariant" is obvious. The contribution is proving it works for VLA transfer.

**Uncertainties:**
1. Is joint-torque-based contact estimation accurate enough?
2. Does the latent space preserve enough information for precise manipulation?
3. Does ManiSkill3 provide comparable difficulty across robots?

**What would make this foundational:**
If the dynamics model predicts object motion for a robot it has NEVER seen, with only a new contact estimator — that's "physics transfers across morphologies."

---

## 8. Related Work

### Cross-Embodiment VLA Models
RT-X (2023), Octo (2024), OpenVLA (2024), π₀ (2024), X-VLA (2025), UniAct (CVPR 2025), OPFA (ICRA 2026), UniVLA (RSS 2025), FAST (2025), π₀.₇ (2026), GR00T N1 (2025)

### Cross-Embodiment Dynamics
He et al. (2025) — particle dynamics for hands. AnyCar (2024) — wheeled vehicles. Lagrangian GNN (2022) — articulated bodies. TraceGen (2025), LAC-WM (2025), Motus (2025) — video-based world models. RoboPack (2024) — single-robot tactile dynamics.

### Cross-Embodiment Policies
Ai et al. (2025) — locomotion scaling laws. Yang, Finn (2026) — data analogies. Wei et al. (2026) — canonical hand representations. Siebenborn (2026) — equivariant bimanual. XHugWBC (2026) — cross-humanoid. GCNT (2025) — graph-based morphology-agnostic.
