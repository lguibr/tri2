# File: agent/replay_buffer/uniform_buffer.py
import random
import numpy as np
from collections import deque
from typing import Deque, Tuple, Optional, Any, Dict, Union, List  # Added List
from .base_buffer import ReplayBufferBase

# --- MODIFIED: Import specific StateType ---
from environment.game_state import StateType  # Use the Dict type

# --- END MODIFIED ---
from utils.types import Transition, ActionType, NumpyBatch, NumpyNStepBatch
from utils.helpers import save_object, load_object


class UniformReplayBuffer(ReplayBufferBase):
    """Standard uniform experience replay buffer."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: StateType,  # State is now a Dict
        action: ActionType,
        reward: float,
        next_state: StateType,  # Next state is now a Dict
        done: bool,
        **kwargs,
    ):
        n_step_discount = kwargs.get("n_step_discount")
        # Store the dictionary state directly in the Transition
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            n_step_discount=n_step_discount,
        )
        self.buffer.append(transition)

    # --- MODIFIED: Sample unpacks dictionary states ---
    def sample(self, batch_size: int) -> Optional[Union[NumpyBatch, NumpyNStepBatch]]:
        if len(self.buffer) < batch_size:
            return None

        batch_transitions: List[Transition] = random.sample(self.buffer, batch_size)
        is_n_step = batch_transitions[0].n_step_discount is not None

        # Unpack transitions, keeping states as dicts for now
        states_dicts = [t.state for t in batch_transitions]
        actions_np = np.array([t.action for t in batch_transitions], dtype=np.int64)
        rewards_np = np.array([t.reward for t in batch_transitions], dtype=np.float32)
        next_states_dicts = [t.next_state for t in batch_transitions]
        dones_np = np.array([t.done for t in batch_transitions], dtype=np.float32)

        if is_n_step:
            discounts_np = np.array(
                [t.n_step_discount for t in batch_transitions], dtype=np.float32
            )
            # Return states as list of dicts, agent will handle conversion
            return (
                states_dicts,
                actions_np,
                rewards_np,
                next_states_dicts,
                dones_np,
                discounts_np,
            )
        else:
            # Return states as list of dicts
            return states_dicts, actions_np, rewards_np, next_states_dicts, dones_np

    # --- END MODIFIED ---

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        pass

    def set_beta(self, beta: float):
        pass

    def flush_pending(self):
        pass

    def __len__(self) -> int:
        return len(self.buffer)

    def get_state(self) -> Dict[str, Any]:
        return {"buffer": list(self.buffer)}

    def load_state_from_data(self, state: Dict[str, Any]):
        saved_buffer_list = state.get("buffer", [])
        # Ensure loaded items are Transitions (or handle potential errors)
        valid_transitions = [t for t in saved_buffer_list if isinstance(t, Transition)]
        if len(valid_transitions) != len(saved_buffer_list):
            print(
                f"Warning: Filtered out {len(saved_buffer_list) - len(valid_transitions)} invalid items during buffer load."
            )
        self.buffer = deque(valid_transitions, maxlen=self.capacity)
        print(f"[UniformReplayBuffer] Loaded {len(self.buffer)} transitions.")

    def save_state(self, filepath: str):
        state = self.get_state()
        save_object(state, filepath)

    def load_state(self, filepath: str):
        state = load_object(filepath)
        self.load_state_from_data(state)
