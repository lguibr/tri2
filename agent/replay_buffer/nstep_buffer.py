# File: agent/replay_buffer/nstep_buffer.py
from collections import deque
import numpy as np
from typing import Deque, Tuple, Optional, Any, Dict, List, Union
from .base_buffer import ReplayBufferBase
from environment.game_state import StateType  # Use the Dict type
from utils.types import Transition, ActionType, NumpyBatch, NumpyNStepBatch
from utils.helpers import save_object, load_object


class NStepBufferWrapper(ReplayBufferBase):
    """Wraps another buffer to implement N-step returns."""

    def __init__(self, wrapped_buffer: ReplayBufferBase, n_step: int, gamma: float):
        super().__init__(wrapped_buffer.capacity)
        if n_step <= 0:
            raise ValueError("N-step must be positive")
        self.wrapped_buffer = wrapped_buffer
        self.n_step = n_step
        self.gamma = gamma
        # Deque stores (state_dict, action, reward, next_state_dict, done) tuples
        self.n_step_deque: Deque[
            Tuple[StateType, ActionType, float, StateType, bool]
        ] = deque(maxlen=n_step)
        print(f"[NStepWrapper] Initialized with n={n_step}, gamma={gamma}")

    def _calculate_n_step_transition(
        self, current_deque_list: List[Tuple]
    ) -> Optional[Transition]:
        """Calculates the N-step return from a list copy of the deque."""
        if not current_deque_list:
            return None

        n_step_reward = 0.0
        discount_accum = 1.0
        effective_n = len(current_deque_list)
        state_0, action_0 = (
            current_deque_list[0][0],
            current_deque_list[0][1],
        )

        for i in range(effective_n):
            _s, _a, reward_i, next_state_i, done_i = current_deque_list[i]
            n_step_reward += discount_accum * reward_i
            if done_i:
                # Episode terminated within N steps
                n_step_next_state = next_state_i  # State after the terminal action
                n_step_done = True
                n_step_discount = self.gamma ** (i + 1)
                return Transition(
                    state_0,
                    action_0,
                    n_step_reward,
                    n_step_next_state,
                    n_step_done,
                    n_step_discount,
                )
            discount_accum *= self.gamma

        # Loop completed without terminal state
        n_step_next_state = current_deque_list[-1][3]  # next_state from Nth transition
        n_step_done = current_deque_list[-1][4]  # done flag from Nth transition
        n_step_discount = self.gamma**effective_n
        return Transition(
            state_0,
            action_0,
            n_step_reward,
            n_step_next_state,
            n_step_done,
            n_step_discount,
        )

    def push(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool,
    ):
        self.n_step_deque.append((state, action, reward, next_state, done))

        if len(self.n_step_deque) < self.n_step:
            if done:
                self._flush_on_done()
            return

        # Deque has N items, calculate N-step transition starting from the oldest
        # Pass a copy to avoid modification if _flush_on_done is called concurrently
        n_step_transition = self._calculate_n_step_transition(list(self.n_step_deque))
        if n_step_transition:
            self.wrapped_buffer.push(
                state=n_step_transition.state,
                action=n_step_transition.action,
                reward=n_step_transition.reward,
                next_state=n_step_transition.next_state,
                done=n_step_transition.done,
                n_step_discount=n_step_transition.n_step_discount,
            )

        if done:
            self._flush_on_done()

    def _flush_on_done(self):
        """Processes remaining partial transitions when an episode ends."""
        temp_deque = list(self.n_step_deque)
        while len(temp_deque) > 1:
            n_step_transition = self._calculate_n_step_transition(temp_deque)
            if n_step_transition:
                self.wrapped_buffer.push(
                    state=n_step_transition.state,
                    action=n_step_transition.action,
                    reward=n_step_transition.reward,
                    next_state=n_step_transition.next_state,
                    done=n_step_transition.done,
                    n_step_discount=n_step_transition.n_step_discount,
                )
            temp_deque.pop(0)
        self.n_step_deque.clear()

    def sample(self, batch_size: int) -> Any:
        # Sampling is delegated, wrapped buffer returns N-step or 1-step batch
        return self.wrapped_buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.wrapped_buffer.update_priorities(indices, priorities)

    def set_beta(self, beta: float):
        if hasattr(self.wrapped_buffer, "set_beta"):
            self.wrapped_buffer.set_beta(beta)

    def __len__(self) -> int:
        return len(self.wrapped_buffer)

    def flush_pending(self):
        print(
            f"[NStepWrapper] Flushing {len(self.n_step_deque)} pending transitions..."
        )
        temp_deque = list(self.n_step_deque)
        while len(temp_deque) > 0:
            n_step_transition = self._calculate_n_step_transition(temp_deque)
            if n_step_transition:
                self.wrapped_buffer.push(
                    state=n_step_transition.state,
                    action=n_step_transition.action,
                    reward=n_step_transition.reward,
                    next_state=n_step_transition.next_state,
                    done=n_step_transition.done,
                    n_step_discount=n_step_transition.n_step_discount,
                )
            temp_deque.pop(0)
        self.n_step_deque.clear()
        if hasattr(self.wrapped_buffer, "flush_pending"):
            self.wrapped_buffer.flush_pending()
        print("[NStepWrapper] Flushing complete.")

    def get_state(self) -> Dict[str, Any]:
        return {
            "n_step_deque": list(self.n_step_deque),
            "wrapped_buffer_state": self.wrapped_buffer.get_state(),
        }

    def load_state_from_data(self, state: Dict[str, Any]):
        pending_deque_list = state.get("n_step_deque", [])
        valid_pending = [
            t for t in pending_deque_list if isinstance(t, tuple) and len(t) == 5
        ]
        if len(valid_pending) != len(pending_deque_list):
            print(
                f"Warning: Filtered {len(pending_deque_list) - len(valid_pending)} invalid items during NStep deque load."
            )
        self.n_step_deque = deque(valid_pending, maxlen=self.n_step)
        print(f"[NStepWrapper] Loaded {len(self.n_step_deque)} pending transitions.")

        wrapped_state = state.get("wrapped_buffer_state")
        if wrapped_state:
            self.wrapped_buffer.load_state_from_data(wrapped_state)
        else:
            print("[NStepWrapper] Warning: No wrapped buffer state found during load.")

    def save_state(self, filepath: str):
        save_object(self.get_state(), filepath)

    def load_state(self, filepath: str):
        try:
            state_data = load_object(filepath)
            self.load_state_from_data(state_data)
        except Exception as e:
            print(f"[NStepWrapper] Load failed: {e}. Starting empty.")
            self.n_step_deque.clear()
            # Recreate the wrapped buffer if load fails to ensure clean state
            if hasattr(self.wrapped_buffer, "__class__"):
                try:
                    self.wrapped_buffer = self.wrapped_buffer.__class__(
                        capacity=self.wrapped_buffer.capacity
                    )
                    print(
                        "[NStepWrapper] Recreated empty wrapped buffer after load failure."
                    )
                except Exception as recreat_e:
                    print(
                        f"[NStepWrapper] Failed to recreate wrapped buffer: {recreat_e}"
                    )
