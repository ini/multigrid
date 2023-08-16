from __future__ import annotations

import os

from pathlib import Path
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from typing import Callable



def can_use_gpu() -> bool:
    """
    Return whether or not GPU training is available.
    """
    try:
        _, tf, _ = try_import_tf()
        return tf.test.is_gpu_available()
    except:
        pass

    try:
        torch, _ = try_import_torch()
        return torch.cuda.is_available()
    except:
        pass

    return False

def find_checkpoint_dir(search_dir: Path | str | None) -> Path | None:
    """
    Recursively search for RLlib checkpoints within the given directory.

    If more than one is found, returns the most recently modified checkpoint directory.

    Parameters
    ----------
    search_dir : Path or str
        The directory to search for checkpoints within
    """
    try:
        checkpoints = Path(search_dir).expanduser().glob('**/*.is_checkpoint')
        if checkpoints:
            return sorted(checkpoints, key=os.path.getmtime)[-1].parent
    except:
        return None

def get_policy_mapping_fn(
    checkpoint_dir: Path | str | None, num_agents: int) -> Callable:
    """
    Create policy mapping function from saved policies in checkpoint directory.
    Maps agent i to the (i % num_policies)-th policy.

    Parameters
    ----------
    checkpoint_dir : Path or str
        The checkpoint directory to load policies from
    num_agents : int
        The number of agents in the environment
    """
    try:
        policies = sorted([
            path for path in (checkpoint_dir / 'policies').iterdir() if path.is_dir()])

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return policies[agent_id % len(policies)].name

        print('Loading policies from:', checkpoint_dir)
        for agent_id in range(num_agents):
            print('Agent ID:', agent_id, 'Policy ID:', policy_mapping_fn(agent_id))

        return policy_mapping_fn

    except:
        return lambda agent_id, *args, **kwargs: f'policy_{agent_id}'
