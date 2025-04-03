from collections import deque
import numpy as np
from typing import Deque, Tuple, Optional, Any, Dict  
from .base_buffer import ReplayBufferBase
from utils.types import Transition, StateType, ActionType
from utils.helpers import save_object, load_object 


class NStepBufferWrapper(ReplayBufferBase):
    """
    Wraps another buffer to implement N-step returns.
    Calculates N-step transitions (s, a, R_n, s_n, done_n, gamma^k)
    and pushes them to the wrapped buffer.
    """

    def __init__(self, wrapped_buffer: ReplayBufferBase, n_step: int, gamma: float):
        super().__init__(wrapped_buffer.capacity)
        if n_step <= 0:
            raise ValueError("N-step must be positive")
        self.wrapped_buffer = wrapped_buffer
        self.n_step = n_step
        self.gamma = gamma
        # Stores raw (s, a, r, ns, d) tuples from the environment steps
        self.n_step_deque: Deque[
            Tuple[StateType, ActionType, float, StateType, bool]
        ] = deque(maxlen=n_step)

    def _process_n_step_buffer(self) -> Optional[Tuple]:
        """
        Calculates N-step info from the oldest entry in the deque.
        Returns: (state_0, action_0, n_step_reward, n_step_next_state, n_step_done, n_step_discount)
        where n_step_discount is gamma^k used for the target Q(n_step_next_state).
        """
        if len(self.n_step_deque) < self.n_step:
            return None  # Not enough steps yet for a full N-step transition

        n_step_reward = 0.0
        discount_accum = 1.0  # Tracks gamma^i for reward accumulation

        # Extract initial state and action
        state_0, action_0 = self.n_step_deque[0][0], self.n_step_deque[0][1]

        # Iterate through the N steps (or until done)
        for i in range(self.n_step):
            s, a, r, ns, d = self.n_step_deque[i]
            n_step_reward += discount_accum * r

            if d:  # Episode terminated within N steps at step i+1
                n_step_next_state = ns  # The terminal state
                n_step_done = True
                # The discount for Q(terminal) is 0
                n_step_discount = 0.0
                # Remove processed transitions from the deque up to and including this one
                # This is tricky with deque's maxlen. We need to pop from left after processing.
                # Let's return the result and pop ONE later in the push/flush logic.
                return (
                    state_0,
                    action_0,
                    n_step_reward,
                    n_step_next_state,
                    n_step_done,
                    n_step_discount,
                )

            # Accumulate discount factor for the reward of the *next* step
            discount_accum *= self.gamma

        # If loop completes without early termination:
        n_step_next_state = self.n_step_deque[-1][3]  # State after N steps
        n_step_done = self.n_step_deque[-1][4]  # Done flag after N steps
        # Discount factor for Q(s_N) is gamma^N
        n_step_discount = self.gamma**self.n_step

        return (
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
        """Adds raw transition, processes N-step if possible, pushes to wrapped buffer."""
        # Ensure inputs are numpy arrays if needed by wrapped buffer (should be handled upstream)
        self.n_step_deque.append((state, action, reward, next_state, done))

        # If deque is now full (or more), process the oldest item
        if len(self.n_step_deque) >= self.n_step:
            processed_info = self._process_n_step_buffer()
            if processed_info:
                s, a, r_n, ns_n, d_n, gamma_n = processed_info
                # Push the calculated N-step transition to the underlying buffer
                self.wrapped_buffer.push(
                    state=s,
                    action=a,
                    reward=r_n,  # N-step cumulative reward
                    next_state=ns_n,  # State after N steps (or terminal state)
                    done=d_n,  # Done flag after N steps (or terminal)
                    n_step_discount=gamma_n,  # gamma^k for Q(ns_n)
                )
                # Since we processed the oldest, remove it (implicitly handled by maxlen if full,
                # but explicit pop might be needed if _process returns early and deque was already full?)
                # Let's rely on maxlen behavior and the flush logic for termination.

        # <<< MODIFIED >>> Flush logic on terminal state detection
        if done:
            # When an episode ends, flush all remaining partial transitions from the deque
            # Process from oldest to newest until deque is shorter than N
            while len(self.n_step_deque) >= 1:
                # Temporarily adjust n_step for processing remaining items
                current_n = len(self.n_step_deque)
                # Create a temporary deque for processing
                temp_deque = deque(list(self.n_step_deque), maxlen=current_n)

                # Process this partial transition (similar logic to _process)
                n_step_reward = 0.0
                discount_accum = 1.0
                state_0, action_0 = temp_deque[0][0], temp_deque[0][1]
                processed_steps = 0

                for i in range(current_n):
                    s, a, r, ns, d = temp_deque[i]
                    n_step_reward += discount_accum * r
                    processed_steps += 1
                    if d:
                        n_step_next_state = ns
                        n_step_done = True
                        n_step_discount = 0.0
                        break  # Found terminal state
                    discount_accum *= self.gamma
                else:  # Loop finished without break (shouldn't happen if done=True was passed to push)
                    # This case means the 'done' transition is the last one in the partial deque
                    n_step_next_state = temp_deque[-1][3]
                    n_step_done = temp_deque[-1][4]  # Should be True
                    n_step_discount = 0.0  # Because it ended

                # Push the processed partial N-step transition
                self.wrapped_buffer.push(
                    state=state_0,
                    action=action_0,
                    reward=n_step_reward,
                    next_state=n_step_next_state,
                    done=n_step_done,
                    n_step_discount=n_step_discount,
                )

                # Remove the oldest item from the main deque now that it's processed
                self.n_step_deque.popleft()

    def sample(self, batch_size: int) -> Any:
        """Samples directly from the wrapped buffer."""
        return self.wrapped_buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.wrapped_buffer.update_priorities(indices, priorities)

    def set_beta(self, beta: float):
        self.wrapped_buffer.set_beta(beta)

    def flush_pending(self):
        """Processes and pushes any remaining transitions before exit/save."""
        print(
            f"[NStepWrapper] Flushing {len(self.n_step_deque)} pending transitions on cleanup."
        )
        # Use the same logic as terminal state handling in push
        while len(self.n_step_deque) >= 1:
            current_n = len(self.n_step_deque)
            temp_deque = deque(list(self.n_step_deque), maxlen=current_n)

            n_step_reward = 0.0
            discount_accum = 1.0
            state_0, action_0 = temp_deque[0][0], temp_deque[0][1]

            for i in range(current_n):
                s, a, r, ns, d = temp_deque[i]
                n_step_reward += discount_accum * r
                if d:
                    n_step_next_state = ns
                    n_step_done = True
                    n_step_discount = 0.0
                    break
                discount_accum *= self.gamma
            else:  # Reached end of deque without termination
                n_step_next_state = temp_deque[-1][3]
                n_step_done = temp_deque[-1][4]
                # Discount is gamma^k where k is the number of steps processed
                n_step_discount = (
                    self.gamma**current_n
                )  # Should this be gamma**(current_n)? Yes.

            # Push the flushed transition
            self.wrapped_buffer.push(
                state=state_0,
                action=action_0,
                reward=n_step_reward,
                next_state=n_step_next_state,
                done=n_step_done,
                n_step_discount=n_step_discount,
            )
            # Remove oldest from main deque
            self.n_step_deque.popleft()

        # Also flush the underlying buffer if it has its own mechanism
        self.wrapped_buffer.flush_pending()

    def __len__(self) -> int:
        return len(self.wrapped_buffer)

    def get_state(self) -> Dict[str, Any]:
        """Return state for saving."""
        return {
            "n_step_deque": list(self.n_step_deque),  # Save pending raw transitions
            "wrapped_buffer_state": self.wrapped_buffer.get_state(),  # Save wrapped buffer's state
        }

    def load_state_from_data(self, state: Dict[str, Any]):
        """Load state from dictionary."""
        # Load pending transitions
        saved_deque_list = state.get("n_step_deque", [])
        self.n_step_deque = deque(saved_deque_list, maxlen=self.n_step)
        print(f"[NStepWrapper] Loaded {len(self.n_step_deque)} pending transitions.")

        # Load wrapped buffer state
        wrapped_state = state.get("wrapped_buffer_state")
        if wrapped_state is not None:
            self.wrapped_buffer.load_state_from_data(wrapped_state)
        else:
            print(
                "[NStepWrapper] Warning: No wrapped buffer state found in saved data."
            )

    def save_state(self, filepath: str):
        """Save buffer state to file."""
        state = self.get_state()
        save_object(state, filepath)

    def load_state(self, filepath: str):
        """Load buffer state from file."""
        state = load_object(filepath)
        self.load_state_from_data(state)
