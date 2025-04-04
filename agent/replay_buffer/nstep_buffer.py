# File: agent/replay_buffer/nstep_buffer.py
# (No structural changes, cleanup comments)
from collections import deque
import numpy as np
from typing import Deque, Tuple, Optional, Any, Dict, List
from .base_buffer import ReplayBufferBase
from utils.types import Transition, StateType, ActionType
from utils.helpers import save_object, load_object


class NStepBufferWrapper(ReplayBufferBase):
    """
    Wraps another buffer to implement N-step returns.
    Calculates N-step transitions and pushes them to the wrapped buffer.
    """

    def __init__(self, wrapped_buffer: ReplayBufferBase, n_step: int, gamma: float):
        super().__init__(wrapped_buffer.capacity)  # Capacity managed by wrapped buffer
        if n_step <= 0:
            raise ValueError("N-step must be positive")
        self.wrapped_buffer = wrapped_buffer
        self.n_step = n_step
        self.gamma = gamma
        # Temporary deque for raw (s, a, r, ns, d) tuples
        self.n_step_deque: Deque[
            Tuple[StateType, ActionType, float, StateType, bool]
        ] = deque(maxlen=n_step)

    def _calculate_n_step_transition(
        self, current_deque_list: List[Tuple]
    ) -> Optional[Transition]:
        """Calculates the N-step return from a list copy of the deque."""
        if not current_deque_list:
            return None

        n_step_reward = 0.0
        discount_accum = 1.0
        effective_n = len(current_deque_list)

        state_0, action_0 = current_deque_list[0][0], current_deque_list[0][1]

        for i in range(effective_n):
            s, a, r, ns, d = current_deque_list[i]
            n_step_reward += discount_accum * r

            if d:  # Episode terminated within these N steps
                n_step_next_state = ns  # Terminal state
                n_step_done = True
                n_step_discount = self.gamma ** (
                    i + 1
                )  # Discount for Q(s_terminal) is effectively 0 later, but store factor
                return Transition(
                    state=state_0,
                    action=action_0,
                    reward=n_step_reward,
                    next_state=n_step_next_state,
                    done=n_step_done,
                    n_step_discount=n_step_discount,
                )

            discount_accum *= self.gamma

        # Loop completed without terminal state
        n_step_next_state = current_deque_list[-1][3]  # next_state from Nth transition
        n_step_done = current_deque_list[-1][4]  # done flag from Nth transition
        n_step_discount = (
            self.gamma**effective_n
        )  # Discount factor for Q(s_N) is gamma^N

        return Transition(
            state=state_0,
            action=action_0,
            reward=n_step_reward,
            next_state=n_step_next_state,
            done=n_step_done,
            n_step_discount=n_step_discount,
        )

    def push(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool,
    ):
        """Adds raw transition, processes N-step if possible, pushes to wrapped buffer."""
        current_transition = (state, action, reward, next_state, done)
        self.n_step_deque.append(current_transition)

        if len(self.n_step_deque) < self.n_step:
            if done:
                self._flush_on_done()  # Process partial if episode ends early
            return  # Wait for more steps

        # Deque has N items, calculate N-step transition from the oldest start
        n_step_transition = self._calculate_n_step_transition(list(self.n_step_deque))

        if n_step_transition:
            self.wrapped_buffer.push(
                state=n_step_transition.state,
                action=n_step_transition.action,
                reward=n_step_transition.reward,
                next_state=n_step_transition.next_state,
                done=n_step_transition.done,
                n_step_discount=n_step_transition.n_step_discount,  # Pass calculated discount
            )

        if done:  # If the *newly added* transition was terminal, flush remaining starts
            self._flush_on_done()

    def _flush_on_done(self):
        """Processes remaining partial transitions when an episode ends."""
        # The deque contains transitions leading up to 'done'.
        # We need to process transitions starting *after* the initial state
        # of the first transition processed in the last `push` call.
        temp_deque = list(self.n_step_deque)
        while len(temp_deque) > 1:  # Stop when only the 'done' transition remains
            temp_deque.pop(0)  # Remove the already processed starting state
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
        self.n_step_deque.clear()  # Clear after flushing

    def sample(self, batch_size: int) -> Any:
        return self.wrapped_buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.wrapped_buffer.update_priorities(indices, priorities)

    def set_beta(self, beta: float):
        if hasattr(self.wrapped_buffer, "set_beta"):
            self.wrapped_buffer.set_beta(beta)

    def flush_pending(self):
        """Processes and pushes any remaining transitions before exit/save."""
        print(
            f"[NStepWrapper] Flushing {len(self.n_step_deque)} pending transitions on cleanup."
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

    def __len__(self) -> int:
        # Length is the number of processed N-step transitions in wrapped buffer
        return len(self.wrapped_buffer)

    def get_state(self) -> Dict[str, Any]:
        """Return state for saving."""
        return {
            "n_step_deque": list(self.n_step_deque),  # Pending raw transitions
            "wrapped_buffer_state": self.wrapped_buffer.get_state(),
        }

    def load_state_from_data(self, state: Dict[str, Any]):
        """Load state from dictionary."""
        saved_deque_list = state.get("n_step_deque", [])
        self.n_step_deque = deque(saved_deque_list, maxlen=self.n_step)
        print(f"[NStepWrapper] Loaded {len(self.n_step_deque)} pending transitions.")

        wrapped_state = state.get("wrapped_buffer_state")
        if wrapped_state is not None:
            self.wrapped_buffer.load_state_from_data(wrapped_state)
        else:
            print(
                "[NStepWrapper] Warning: No wrapped buffer state found in saved data."
            )

    def save_state(self, filepath: str):
        state = self.get_state()
        save_object(state, filepath)

    def load_state(self, filepath: str):
        try:
            state = load_object(filepath)
            self.load_state_from_data(state)
        except FileNotFoundError:
            print(
                f"[NStepWrapper] Load failed: File not found at {filepath}. Starting empty."
            )
        except Exception as e:
            print(f"[NStepWrapper] Load failed: {e}. Starting empty.")
