import random
import numpy as np
from collections import deque
from typing import Deque, Tuple, Optional, Any, Dict, Union 
from .base_buffer import ReplayBufferBase
from utils.types import (
    Transition,
    StateType,
    ActionType,
    NumpyBatch,
    NumpyNStepBatch,
)  
from utils.helpers import save_object, load_object  


class UniformReplayBuffer(ReplayBufferBase):
    """Standard uniform experience replay buffer."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool,
        **kwargs,  # Accept potential n_step_discount from NStepWrapper
    ):
        # NStepWrapper passes n_step processed values (r_n, ns_n, d_n, gamma_n)
        # Store them directly in the Transition object
        n_step_discount = kwargs.get("n_step_discount")  # Get optional discount
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            n_step_discount=n_step_discount,  # Store the discount
        )
        self.buffer.append(transition)

    def sample(
        self, batch_size: int
    ) -> Optional[
        Union[NumpyBatch, NumpyNStepBatch]
    ]: 
        if len(self.buffer) < batch_size:
            return None

        batch_transitions = random.sample(self.buffer, batch_size)

        # Check if the first sampled transition has n_step_discount to determine batch type
        # Assumes all transitions in buffer are consistently 1-step or N-step if NStepWrapper is used
        is_n_step = batch_transitions[0].n_step_discount is not None

        if is_n_step:
            # Unpack N-step fields: state, action, n_reward, n_next_state, n_done, n_discount
            s, a, rn, nsn, dn, gamma_n = zip(
                *[
                    (
                        t.state,
                        t.action,
                        t.reward,
                        t.next_state,
                        t.done,
                        t.n_step_discount,
                    )
                    for t in batch_transitions
                ]
            )
            states_np = np.array(s, dtype=np.float32)
            actions_np = np.array(a, dtype=np.int64)
            rewards_np = np.array(rn, dtype=np.float32)
            next_states_np = np.array(nsn, dtype=np.float32)
            dones_np = np.array(dn, dtype=np.float32)
            discounts_np = np.array(
                gamma_n, dtype=np.float32
            )  # gamma^k discount factor
            return (
                states_np,
                actions_np,
                rewards_np,
                next_states_np,
                dones_np,
                discounts_np,
            )
        else:
            # Unpack 1-step fields: state, action, reward, next_state, done
            s, a, r, ns, d = zip(
                *[
                    (t.state, t.action, t.reward, t.next_state, t.done)
                    for t in batch_transitions
                ]
            )
            states_np = np.array(s, dtype=np.float32)
            actions_np = np.array(a, dtype=np.int64)
            rewards_np = np.array(r, dtype=np.float32)
            next_states_np = np.array(ns, dtype=np.float32)
            dones_np = np.array(d, dtype=np.float32)
            return states_np, actions_np, rewards_np, next_states_np, dones_np

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        pass  # No-op

    def set_beta(self, beta: float):
        pass  # No-op

    def flush_pending(self):
        pass  # No-op

    def __len__(self) -> int:
        return len(self.buffer)

    # <<< NEW >>> Save/Load state using pickle/cloudpickle
    def get_state(self) -> Dict[str, Any]:
        """Return state for pickling."""
        return {
            "buffer": list(self.buffer)
        }  # Convert deque to list for robust pickling

    def load_state_from_data(self, state: Dict[str, Any]):
        """Load state from dictionary."""
        saved_buffer_list = state.get("buffer", [])
        # Ensure capacity consistency? For now, just load what fits.
        # Recreate deque with potentially different capacity? Let's keep capacity.
        self.buffer = deque(saved_buffer_list, maxlen=self.capacity)
        print(f"[UniformReplayBuffer] Loaded {len(self.buffer)} transitions.")

    def save_state(self, filepath: str):
        """Save buffer state to file."""
        state = self.get_state()
        save_object(state, filepath)

    def load_state(self, filepath: str):
        """Load buffer state from file."""
        state = load_object(filepath)
        self.load_state_from_data(state)
