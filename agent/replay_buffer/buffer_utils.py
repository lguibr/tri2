from config import BufferConfig, DQNConfig
from .base_buffer import ReplayBufferBase
from .uniform_buffer import UniformReplayBuffer
from .prioritized_buffer import PrioritizedReplayBuffer
from .nstep_buffer import NStepBufferWrapper


def create_replay_buffer(
    config: BufferConfig, dqn_config: DQNConfig
) -> ReplayBufferBase:
    """Factory function to create the appropriate replay buffer based on config."""

    print("[BufferFactory] Creating replay buffer...")
    print(f"  Capacity: {config.REPLAY_BUFFER_SIZE}")
    print(f"  Use PER: {config.USE_PER}")
    print(f"  Use N-Step: {config.USE_N_STEP} (N={config.N_STEP})")

    if config.USE_PER:
        core_buffer = PrioritizedReplayBuffer(
            capacity=config.REPLAY_BUFFER_SIZE,
            alpha=config.PER_ALPHA,
            epsilon=config.PER_EPSILON,
        )
        print(
            f"  Type: Prioritized (alpha={config.PER_ALPHA}, eps={config.PER_EPSILON})"
        )
    else:
        core_buffer = UniformReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
        print("  Type: Uniform")

    if config.USE_N_STEP and config.N_STEP > 1:
        final_buffer = NStepBufferWrapper(
            wrapped_buffer=core_buffer,
            n_step=config.N_STEP,
            gamma=dqn_config.GAMMA,
        )
        print(
            f"  Wrapped with: NStepBufferWrapper (N={config.N_STEP}, gamma={dqn_config.GAMMA})"
        )
    else:
        final_buffer = core_buffer

    print(f"[BufferFactory] Final buffer type: {type(final_buffer).__name__}")
    return final_buffer
