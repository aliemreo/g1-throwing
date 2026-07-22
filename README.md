# G1 Whole-Body Throwing

RL controller that throws a ball at a commanded target using the Unitree G1
in MuJoCo. The policy outputs joint residuals around the standing pose plus
a continuous release signal.

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

## Train

    python train_ppo.py --easy --envs 12 --steps 3000000
    tensorboard --logdir runs/throw_v1/tb

Watch `throw/mean_miss`, not reward. Reward is not comparable across episodes
because distant targets give the shaping term more room to accumulate.

Set --envs to your physical core count.

## Evaluate

    python eval_policy.py --run throw_v1 --episodes 50
    python eval_policy.py --run throw_v1 --view
