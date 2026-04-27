from __future__ import annotations

from collections import deque

import numpy as np


class GradReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.capacity = int(capacity)
        self._storage = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self._storage)

    def add(self, state, action_grad, action=None, step=None, time=None, loss=None):
        transition = {
            "state": np.asarray(state, dtype=np.float32).copy(),
            "action_grad": np.asarray(action_grad, dtype=np.float32).copy(),
            "action": None if action is None else np.asarray(action, dtype=np.float32).copy(),
            "step": None if step is None else int(step),
            "time": None if time is None else float(time),
            "loss": None if loss is None else float(loss),
        }
        self._storage.append(transition)

    def extend(self, transitions):
        for transition in transitions:
            self.add(**transition)

    def sample(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(self._storage) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        batch_size = int(batch_size)
        if batch_size <= len(self._storage):
            indices = np.random.choice(len(self._storage), size=batch_size, replace=False)
        else:
            indices = np.random.randint(0, len(self._storage), size=batch_size)

        batch = [self._storage[idx] for idx in indices]
        states = np.stack([item["state"] for item in batch], axis=0)
        action_grads = np.stack([item["action_grad"] for item in batch], axis=0)
        actions = None
        if batch[0]["action"] is not None:
            actions = np.stack([item["action"] for item in batch], axis=0)

        return {
            "state": states,
            "action_grad": action_grads,
            "action": actions,
            "step": np.asarray([item["step"] for item in batch], dtype=np.int64),
            "time": np.asarray([item["time"] for item in batch], dtype=np.float32),
            "loss": np.asarray([item["loss"] for item in batch], dtype=np.float32),
        }