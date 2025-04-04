# File: agent/replay_buffer/buffer_utils.py
from config import BufferConfig, DQNConfig
from .base_buffer import ReplayBufferBase
from .uniform_buffer import UniformReplayBuffer
from .prioritized_buffer import PrioritizedReplayBuffer  # Import PER
from .nstep_buffer import NStepBufferWrapper


def create_replay_buffer(
    config: BufferConfig, dqn_config: DQNConfig
) -> ReplayBufferBase:
    """Factory function to create the replay buffer based on configuration."""

    print("[BufferFactory] Creating replay buffer...")
    print(f"  Type: {'Prioritized' if config.USE_PER else 'Uniform'}")
    print(f"  Capacity: {config.REPLAY_BUFFER_SIZE/1e6:.1f}M")

    # --- MODIFIED: Instantiate core buffer based on USE_PER ---
    if config.USE_PER:
        print(
            f"  PER alpha={config.PER_ALPHA}, eps={config.PER_EPSILON}, beta_start={config.PER_BETA_START}"
        )
        core_buffer = PrioritizedReplayBuffer(
            capacity=config.REPLAY_BUFFER_SIZE,
            alpha=config.PER_ALPHA,
            epsilon=config.PER_EPSILON,
        )
    else:
        core_buffer = UniformReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
    # --- END MODIFIED ---

    if config.USE_N_STEP and config.N_STEP > 1:
        print(
            f"  N-Step Wrapper: Enabled (N={config.N_STEP}, gamma={dqn_config.GAMMA})"
        )
        final_buffer = NStepBufferWrapper(
            wrapped_buffer=core_buffer,
            n_step=config.N_STEP,
            gamma=dqn_config.GAMMA,
        )
    else:
        print(f"  N-Step Wrapper: Disabled")
        final_buffer = core_buffer

    print(f"[BufferFactory] Final buffer type: {type(final_buffer).__name__}")
    return final_buffer
