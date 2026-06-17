import os, glob, sys, json
from datetime import datetime

def read_tb_last_metrics(event_dir, scalar_names):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted(glob.glob(os.path.join(event_dir, 'events.out.*')))
    if not files: return {}
    ea = EventAccumulator(files[-1])
    ea.Reload()
    result = {}
    for name in scalar_names:
        matching = [t for t in ea.Tags().get('scalars', []) if name in t]
        for m in matching:
            events = ea.Scalars(m)
            if events: result[m] = events[-1].value
    return result

ppo_dir = '/home/peterpan/UniLab/logs/rsl_rl_ppo/MyBipedalWalkFlat/2026-06-17_09-10-48_mujoco'
sac_dir = '/home/peterpan/UniLab/logs/fast_sac/MyBipedalWalkFlat/2026-06-17_09-13-27_mujoco'

ppo_names = ['episode_length', 'reward', 'entropy', 'termination_rate', 'linvel_x']
sac_names = ['reward', 'episode_length', 'policy_entropy', 'action_std']

now = datetime.now().strftime('%H:%M:%S')
print(f'=== Training Status @ {now} ===')

ppo = read_tb_last_metrics(ppo_dir, ppo_names)
ppo_ckpts = len(glob.glob(os.path.join(ppo_dir, 'model_*.pt')))
print(f'PPO (iter ~{ppo_ckpts*100}): ', end='')
for k, v in ppo.items():
    short = k.split('/')[-1]
    print(f'{short}={v:.2f} ', end='')
print()

sac = read_tb_last_metrics(sac_dir, sac_names)
sac_ckpts = len(glob.glob(os.path.join(sac_dir, 'model_*.pt')))
print(f'SAC (iter ~{sac_ckpts*1000}): ', end='')
for k, v in sac.items():
    short = k.split('/')[-1]
    print(f'{short}={v:.2f} ', end='')
print()
