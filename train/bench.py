import time
import numpy as np
from env import ThrowEnv

env = ThrowEnv(seed=0)
env.reset()
n = 2000
t0 = time.perf_counter()
for _ in range(n):
    _, _, term, trunc, _ = env.step(np.random.uniform(-1, 1, env.n_act))
    if term or trunc:
        env.reset()
el = time.perf_counter() - t0
print(f"{n/el:8.0f} env steps/sec   ({n*env.decimation/el:.0f} sim steps/sec)")
print(f"1M env steps would take {1e6/(n/el)/60:.1f} minutes single-threaded")