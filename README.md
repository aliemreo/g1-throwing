# G1 TD3 Throwing

TD3 controller that throws a ball to a specified target using the Unitree G1
in MuJoCo. The policy controls the right shoulder pitch/roll, right elbow, and
a continuous release signal. Other joints hold the standing pose.

The reward has exactly one objective: reduce the final XY landing error. It
contains no posture, energy, speed, survival, release, or terminal bonus/penalty.

## Setup

    git clone --recurse-submodules https://github.com/<you>/g1-throwing.git
    cd g1-throwing
    python -m venv .venv
    .venv\Scripts\Activate.ps1        # Windows
    source .venv/bin/activate         # Linux/Mac
    pip install -r requirements.txt

Already cloned without submodules:

    git submodule update --init --recursive

Windows: if venv activation is blocked, run once:

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

## Verify

    cd train
    python check_env.py

Expect a ~0.56 m miss. This replays a hand-tuned swing and serves as the
regression test - if the number changes, something in the physics layer broke.

## Train TD3 for a specified target

    cd train
    python train_td3.py --target 4 0 --steps 1000000
    tensorboard --logdir runs/td3_target_4_0/tb

Watch `throw/mean_miss` in TensorBoard. `best_model.zip` is updated from a
rolling 100-episode mean, while `final.zip` is always written when training
stops. Use `--resume PATH_TO_ZIP` to continue a run.

TD3 is off-policy and uses one MuJoCo environment so replay updates stay
well-defined and reproducible.

## Evaluate

    python eval_td3.py --run td3_target_4_0 --episodes 50
    python eval_td3.py --run td3_target_4_0 --view

The original PPO entry point remains available as `train_ppo.py`.
