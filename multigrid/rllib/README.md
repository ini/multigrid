## Training Policies with RLlib

Train a 2-agent environment using PPO

    python scripts/train.py --algo PPO --framework torch --env MultiGrid-BlockedUnlockPickup-v0 --num-agents 2 --num-timesteps 10000000 --save-dir ~/saved/

Visualize agent behavior:

    python scripts/visualize --algo PPO --framework torch --env MultiGrid-BlockedUnlockPickup-v0 --num-agents 2 --num-timesteps 10000000 --save-dir ~/saved/
