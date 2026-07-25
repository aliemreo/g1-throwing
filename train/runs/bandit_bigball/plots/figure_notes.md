# Figure notes — walking robot that throws to a chosen target (RL)

**Problem.** A humanoid walks forward and must throw a ball to a target placed at
a random distance (3.3–5.0 m) each episode. A reinforcement-learning agent
learns **when to release** the ball; the walking gait and arm swing are fixed and
stable, so releasing earlier/later makes the ball land shorter/farther. Reward
= −(landing error in metres) − 5×(did it fall?). Trained with TD3 (an
actor–critic RL algorithm).

Result: **100% of targets hit within 0.5 m, mean error 0.18 m, 0 falls**, and it
generalises to distances never seen in training (smooth learned map, not
memorisation).

---

## 1. reward_curve.png — is the agent learning?
Average reward per throw over training. It rises sharply during the first ~1,000
steps (the agent discovers release timings that land near the target without
falling), then plateaus. **Takeaway: learning is fast and stable — it converges
to near-optimal in ~1,000 steps and does not degrade afterwards.**

## 2. success_rate.png — how accurate, and how fast?
Fraction of targets landed within 0.5 m, evaluated on 20 distances across the
range. Three shaded phases: **warmup** (replay buffer filling, no weight updates
yet — so it's flat), **learning** (rises from ~55% to 100%), **converged**
(stays at 100%). **Takeaway: once learning starts it reaches a perfect success
rate by ~1,100 steps and holds it.** (The brief dip is the policy first moving
off its initial guess before it organises the target→timing map.)

## 3. fall_count_by_version.png — is it safe?
Each saved policy version is run on 50 random targets; we count falls. **0 falls
at every version.** Because the agent only decides *when* to release and leaves
the (already stable) walk + arm motion untouched, it never destabilises the
robot. **Takeaway: accuracy is achieved with zero loss of balance.**

## 4. tracking_error_dist.png — how good is the final policy, on unseen targets?
Landing-error histogram over **300 fresh random targets not seen in training.**
Every throw is within the 0.5 m band (mean 0.18 m); 0 falls. **Takeaway: the
policy generalises — it isn't memorising training targets.** The spread (rather
than all-near-zero) comes from the finite release-timing resolution: the ball can
only be let go at discrete simulation instants, which quantises the reachable
landing distances. This is a physical limit, not an aiming mistake.

---

### Anticipated jury questions
- *"Why does success jump to 100% so fast?"* The task is a well-posed 1-D control
  problem (target distance → release time) and the RL formulation removes the
  usual credit-assignment difficulty, so it's genuinely easy to learn. The
  histogram on unseen targets confirms it's real, not a fluke.
- *"Do you overfit?"* No — evaluation and the 300-target test use targets drawn
  independently of training; the learned action varies smoothly and monotonically
  with target distance (interpolation), which is the signature of generalisation.
- *"Limits?"* It is trained for this ball, this walk, straight-ahead targets, and
  3.3–5.0 m. Outside that range/direction it would need retraining — that's scope,
  not overfitting.
