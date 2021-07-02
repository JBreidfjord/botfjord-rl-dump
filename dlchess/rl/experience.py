from __future__ import annotations

import h5py
import numpy as np
from sklearn.utils import shuffle


class ExperienceBuffer:
    def __init__(self, states, actions, rewards, advantages):
        self.states: np.ndarray = states
        self.actions: np.ndarray = actions
        self.rewards: np.ndarray = rewards
        self.advantages: np.ndarray = advantages

    def serialize(self, h5file):
        if isinstance(h5file, str):
            h5file = h5py.File(h5file, mode="w")

        h5file.create_group("experience")
        h5file["experience"].create_dataset("states", data=self.states)
        h5file["experience"].create_dataset("actions", data=self.actions)
        h5file["experience"].create_dataset("rewards", data=self.rewards)
        h5file["experience"].create_dataset("advantages", data=self.advantages)

    def to_collector(self):
        collector = ExperienceCollector()
        collector.set_data(
            self.states.tolist(),
            self.actions.tolist(),
            self.rewards.tolist(),
            self.advantages.tolist(),
        )
        return collector


class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_estimated_values = []

    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_estimated_values = []

    def record_decision(self, state, action, estimated_value=0):
        self.current_episode_states.append(state)
        self.current_episode_actions.append(action)
        self.current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward):
        num_states = len(self.current_episode_states)
        self.states.extend(self.current_episode_states)
        self.actions.extend(self.current_episode_actions)
        self.rewards.extend([reward] * num_states)

        for i in range(num_states):
            advantage = reward - self.current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_estimated_values = []

    def reset_episode(self):
        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_estimated_values = []

    def to_buffer(self):
        return ExperienceBuffer(
            states=np.array(self.states),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
            advantages=np.array(self.advantages),
        )

    def set_data(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def is_empty(self) -> bool:
        return (
            self.states == []
            or self.actions == []
            or self.rewards == []
            or self.advantages == []
        )


class ZeroCollector(ExperienceCollector):
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []

        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward: int | None):
        num_states = len(self._current_episode_states)
        self.states.extend(self._current_episode_states)
        self.visit_counts.extend(self._current_episode_visit_counts)
        if reward is not None:
            self.rewards.extend([reward] * num_states)

        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def reset_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def to_buffer(self):
        return ZeroBuffer(
            np.array(self.states), np.array(self.visit_counts), np.array(self.rewards)
        )

    def set_data(self, states, visit_counts, rewards):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards

    def is_empty(self) -> bool:
        return (
            self._current_episode_states == []
            or self._current_episode_visit_counts == []
        )

    def multiply(self, multiplier: int):
        states = self.states
        visit_counts = self.visit_counts
        rewards = self.rewards
        for _ in range(multiplier):
            self.states.extend(states)
            self.visit_counts.extend(visit_counts)
            self.rewards.extend(rewards)

    def shuffle(self):
        self.states, self.visit_counts, self.rewards = shuffle(
            self.states, self.visit_counts, self.rewards
        )


class ZeroBuffer(ExperienceBuffer):
    def __init__(self, states, visit_counts, rewards):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards

    def serialize(self, h5file):
        h5file.create_group("experience")
        h5file["experience"].create_dataset("states", data=self.states)
        h5file["experience"].create_dataset("visit_counts", data=self.visit_counts)
        h5file["experience"].create_dataset("rewards", data=self.rewards)

    def to_collector(self):
        collector = ZeroCollector()
        collector.set_data(
            self.states.tolist(), self.visit_counts.tolist(), self.rewards.tolist()
        )
        return collector

    def remove_draws(self):
        indices = np.nonzero(self.rewards)[0]
        self.states = self.states[indices]
        self.visit_counts = self.visit_counts[indices]
        self.rewards = self.rewards[indices]

    def shuffle(self):
        self.states, self.visit_counts, self.rewards = shuffle(
            self.states, self.visit_counts, self.rewards
        )

    def merge_with(self, other: ZeroBuffer):
        self.states = np.concatenate([self.states, other.states])
        self.visit_counts = np.concatenate([self.visit_counts, other.visit_counts])
        self.rewards = np.concatenate([self.rewards, other.rewards])

    def serialize_split(self, h5files):
        num_split = len(h5files)
        num_states = len(self.states)
        split_size = num_states // num_split
        for i, h5file in enumerate(h5files):
            h5file.create_group("experience")
            h5file["experience"].create_dataset(
                "states", data=self.states[i * split_size : (i + 1) * split_size]
            )
            h5file["experience"].create_dataset(
                "visit_counts",
                data=self.visit_counts[i * split_size : (i + 1) * split_size],
            )
            h5file["experience"].create_dataset(
                "rewards", data=self.rewards[i * split_size : (i + 1) * split_size]
            )

    def serialize_batches(
        self, batch_size: int, filepath, even: bool = True, skip: int = 0
    ):
        """
        Splits data into multiple files based on batch_size\n
        Filepath should be a path including the file name prefix to be saved to\n
        If even is true, files will have a similar sample count <= batch_size;
        If false, files before the last will have exactly batch_size samples
        with the last file containing the rest
        """
        num_states = len(self.states)
        num_splits = (num_states // batch_size) + 1
        if even:
            batch_size = (num_states // num_splits) + 1
        for i in range(skip, num_splits + skip):
            if i - skip == num_splits - 1 and num_states % batch_size == 0:
                continue

            if i - skip == num_splits - 1 and not even:
                h5file = h5py.File(f"{filepath}_partial.h5", mode="w")
                print(
                    f"Partial file will have {len(self.states[(i - skip) * batch_size : (i - skip + 1) * batch_size])} samples"
                )
            else:
                h5file = h5py.File(f"{filepath}_{i}.h5", mode="w")
            i -= skip
            h5file.create_group("experience")
            h5file["experience"].create_dataset(
                "states", data=self.states[i * batch_size : (i + 1) * batch_size]
            )
            h5file["experience"].create_dataset(
                "visit_counts",
                data=self.visit_counts[i * batch_size : (i + 1) * batch_size],
            )
            h5file["experience"].create_dataset(
                "rewards", data=self.rewards[i * batch_size : (i + 1) * batch_size]
            )
            h5file.close()


def load_experience(h5file, zero=False):
    if zero:
        return ZeroBuffer(
            states=np.array(h5file["experience"]["states"]),
            visit_counts=np.array(h5file["experience"]["visit_counts"]),
            rewards=np.array(h5file["experience"]["rewards"]),
        )

    return ExperienceBuffer(
        states=np.array(h5file["experience"]["states"]),
        actions=np.array(h5file["experience"]["actions"]),
        rewards=np.array(h5file["experience"]["rewards"]),
        advantages=np.array(h5file["experience"]["advantages"]),
    )


def combine_experience(collectors, zero=False):
    if zero:
        combined_states = np.concatenate(
            [np.array(c.states) for c in collectors if len(c.states) > 0]
        )
        combined_visit_counts = np.concatenate(
            [np.array(c.visit_counts) for c in collectors if len(c.visit_counts) > 0]
        )
        combined_rewards = np.concatenate(
            [np.array(c.rewards) for c in collectors if len(c.rewards) > 0]
        )

        return ZeroBuffer(combined_states, combined_visit_counts, combined_rewards)

    combined_states = np.concatenate(
        [np.array(c.states) for c in collectors if len(c.states) > 0]
    )
    combined_actions = np.concatenate(
        [np.array(c.actions) for c in collectors if len(c.actions) > 0]
    )
    combined_rewards = np.concatenate(
        [np.array(c.rewards) for c in collectors if len(c.rewards) > 0]
    )
    combined_advantages = np.concatenate(
        [np.array(c.advantages) for c in collectors if len(c.advantages) > 0]
    )

    return ExperienceBuffer(
        combined_states, combined_actions, combined_rewards, combined_advantages
    )
