# Experiment Protocol: Testing Embodiment-Invariant Dynamics

## The Karpathy-Style One-Liner

**Claim:** Object dynamics during pushing are a function of (object state, task-space interaction) — NOT of which robot arm is doing the pushing. If true, a single dynamics model should predict object motion from both robots with zero robot-specific adaptation.

---

## Setup

### Robots

| | Franka Panda | UR5e |
|---|---|---|
| DOF | 7 | 6 |
| Kinematics | Spherical wrist, offset elbow | Nearly-spherical wrist, symmetric elbow |
| Workspace | Dextrous, near-body | Large, symmetric |
| Control | Joint impedance → task-space | Joint position → task-space |

**Why these two:** Different joint count, different kinematics, different workspace shapes — but both terminate in a parallel gripper pressing against a table. The *physics of the push* (friction, object inertia, contact) is identical; only the arm delivering it differs.

### Environment

**ManiSkill3** — `PushCube-v1` task (GPU-parallelized, runs 30K+ FPS)

- Same table, same cube, same friction model
- Task: push cube to a target position
- Randomized: initial cube pose, target pose, robot starting configuration
- Ground-truth object state and contact forces via SAPIEN

### Data Collection (~30 min wall-clock)

Generate **N=2000 trajectories per robot** using a scripted pushing policy:
- Randomize: cube position, push direction, push speed
- Record per timestep: `(object_pose_7D, ee_pose_7D, ee_delta_3D, gripper_state)`
- ~50 timesteps per trajectory → 100k transitions per robot

---

## Architecture

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

All encoders/decoders/dynamics are **shared across robots** — zero robot-specific parameters.

---

## The Three Models

| Model | Training Data | What it tests |
|---|---|---|
| M_Franka | Franka only (2000 traj) | Upper bound on Franka performance |
| M_UR5e | UR5e only (2000 traj) | Upper bound on UR5e performance |
| M_mixed | Both robots (4000 traj, shuffled) | **The hypothesis: shared dynamics works** |

Training: All three use identical architecture, learning rate, epochs. Adam optimizer, lr=1e-3, 100 epochs, batch size 256. ~2 hours total on a single A100.

---

## Evaluation Protocol

### Test Set
- 200 held-out trajectories from each robot (not seen during training)

### Metrics

**1. Cross-robot zero-shot dynamics prediction (the money metric)**
- Take M_mixed, feed it Franka test trajectories
- Predict object_pose at t+1, t+5, t+10, t+20 from t
- Measure: **MSE between predicted and true object pose**

**2. Same-robot baseline (the ceiling)**
- M_Franka on Franka test data
- M_UR5e on UR5e test data

**3. Wrong-robot baseline (the floor)**
- M_UR5e applied to Franka test data
- M_Franka applied to UR5e test data
- This tests whether robot-specific models *fail* on the other robot

**4. Latent dynamics alignment**
- For matched pushing scenarios (same cube pose, same push direction):
- Compare the *direction* of z_{t+1} - z_t across robots
- Metric: cosine similarity of latent transitions

### Aggregate Metric

```
Transfer Ratio = MSE_mixed→Franka / MSE_Franka→Franka
```

---

## Pre-Registered Success Criteria

| Criterion | Threshold | What it means |
|---|---|---|
| Transfer Ratio | ≤ 1.15 | Mixed model ≤15% worse than single-robot |
| Wrong-robot fails | MSE > 2× baseline | Robot-specific models DON'T transfer |
| Latent alignment | cosine > 0.7 | Shared latent space captures same dynamics |

**Pass condition:** ALL three criteria met.
**Fail condition:** ANY criterion fails.

---

## Failure Modes & Mitigations

| Risk | Mitigation |
|---|---|
| Task too easy (trivial dynamics) | Add noise to object physics, vary mass/friction |
| UR5e data too different | Verify scripted policy generates similar push distributions via t-SNE |
| Encoder memorizes robot identity | Check if latent space clusters by robot (t-SNE) |
| ManiSkill3 doesn't support UR5e | Fallback: different Franka control modes |

---

## Compute Budget

| Step | Time | Hardware |
|---|---|---|
| Data generation | ~30 min | GPU (ManiSkill3 parallel envs) |
| Training (3 models) | ~2 hours | Single A100 |
| Evaluation + analysis | ~1 hour | CPU/GPU |
| **Total** | **~4 hours** | **1× A100** |
