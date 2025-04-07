# File: stats/aggregator.py
# File: stats/aggregator.py
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, List
import numpy as np
from config import StatsConfig


class StatsAggregator:
    """
    Handles aggregation and storage of training statistics using deques.
    Calculates rolling averages and tracks best values. Does not perform logging.
    """

    def __init__(
        self,
        avg_windows: List[int] = StatsConfig.STATS_AVG_WINDOW,
        plot_window: int = StatsConfig.PLOT_DATA_WINDOW,
    ):
        if not avg_windows or not all(
            isinstance(w, int) and w > 0 for w in avg_windows
        ):
            print("Warning: Invalid avg_windows list. Using default [100].")
            self.avg_windows = [100]
        else:
            self.avg_windows = sorted(list(set(avg_windows)))

        if plot_window <= 0:
            plot_window = 10000
        self.plot_window = plot_window

        self.summary_avg_window = self.avg_windows[0]

        # --- Deques for Plotting ---
        self.policy_losses: Deque[float] = deque(maxlen=plot_window)
        self.value_losses: Deque[float] = deque(maxlen=plot_window)
        self.entropies: Deque[float] = deque(maxlen=plot_window)
        self.grad_norms: Deque[float] = deque(maxlen=plot_window)
        self.avg_max_qs: Deque[float] = deque(maxlen=plot_window)
        self.episode_scores: Deque[float] = deque(maxlen=plot_window)
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_triangles_cleared: Deque[int] = deque(maxlen=plot_window)
        self.sps_values: Deque[float] = deque(maxlen=plot_window)
        self.buffer_sizes: Deque[int] = deque(maxlen=plot_window)
        self.beta_values: Deque[float] = deque(maxlen=plot_window)
        self.best_rl_score_history: Deque[float] = deque(maxlen=plot_window)
        self.best_game_score_history: Deque[int] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(maxlen=plot_window)
        self.epsilon_values: Deque[float] = deque(maxlen=plot_window)
        # Resource Usage Deques
        self.cpu_usage: Deque[float] = deque(maxlen=plot_window)
        self.memory_usage: Deque[float] = deque(maxlen=plot_window)
        self.gpu_memory_usage_percent: Deque[float] = deque(
            maxlen=plot_window
        )  # Changed key

        # --- Scalar State Variables ---
        self.total_episodes = 0
        self.total_triangles_cleared = 0
        self.current_epsilon: float = 0.0
        self.current_beta: float = 0.0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0
        self.current_sps: float = 0.0
        self.current_lr: float = 0.0
        self.start_time: float = time.time()
        self.training_target_step: int = 0
        # Resource Usage Scalars
        self.current_cpu_usage: float = 0.0
        self.current_memory_usage: float = 0.0
        self.current_gpu_memory_usage_percent: float = 0.0  # Changed key

        # --- Best Value Tracking ---
        self.best_score: float = -float("inf")
        self.previous_best_score: float = -float("inf")
        self.best_score_step: int = 0

        self.best_game_score: float = -float("inf")
        self.previous_best_game_score: float = -float("inf")
        self.best_game_score_step: int = 0

        self.best_value_loss: float = float("inf")
        self.previous_best_value_loss: float = float("inf")
        self.best_value_loss_step: int = 0

        print(
            f"[StatsAggregator] Initialized. Avg Windows: {self.avg_windows}, Plot Window: {self.plot_window}"
        )

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
    ) -> Dict[str, Any]:
        current_step = (
            global_step if global_step is not None else self.current_global_step
        )
        update_info = {"new_best_rl": False, "new_best_game": False}

        self.episode_scores.append(episode_score)
        self.episode_lengths.append(episode_length)
        if game_score is not None:
            self.game_scores.append(game_score)
        if triangles_cleared is not None:
            self.episode_triangles_cleared.append(triangles_cleared)
            self.total_triangles_cleared += triangles_cleared
        self.total_episodes = episode_num

        if episode_score > self.best_score:
            self.previous_best_score = self.best_score
            self.best_score = episode_score
            self.best_score_step = current_step
            update_info["new_best_rl"] = True

        if game_score is not None and game_score > self.best_game_score:
            self.previous_best_game_score = self.best_game_score
            self.best_game_score = float(game_score)
            self.best_game_score_step = current_step
            update_info["new_best_game"] = True

        self.best_rl_score_history.append(self.best_score)
        current_best_game = (
            int(self.best_game_score) if self.best_game_score > -float("inf") else 0
        )
        self.best_game_score_history.append(current_best_game)

        return update_info

    def record_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        g_step = step_data.get("global_step", self.current_global_step)
        if g_step > self.current_global_step:
            self.current_global_step = g_step

        if "training_target_step" in step_data:
            self.training_target_step = step_data["training_target_step"]

        update_info = {"new_best_loss": False}

        if "policy_loss" in step_data and step_data["policy_loss"] is not None:
            loss_val = step_data["policy_loss"]
            if np.isfinite(loss_val):
                self.policy_losses.append(loss_val)
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Policy Loss: {loss_val}"
                )

        if "value_loss" in step_data and step_data["value_loss"] is not None:
            current_value_loss = step_data["value_loss"]
            if np.isfinite(current_value_loss):
                self.value_losses.append(current_value_loss)
                if current_value_loss < self.best_value_loss and g_step > 0:
                    self.previous_best_value_loss = self.best_value_loss
                    self.best_value_loss = current_value_loss
                    self.best_value_loss_step = g_step
                    update_info["new_best_loss"] = True
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Value Loss: {current_value_loss}"
                )

        if "entropy" in step_data and step_data["entropy"] is not None:
            entropy_val = step_data["entropy"]
            if np.isfinite(entropy_val):
                self.entropies.append(entropy_val)
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Entropy: {entropy_val}"
                )

        if "grad_norm" in step_data and step_data["grad_norm"] is not None:
            self.grad_norms.append(step_data["grad_norm"])
        if "avg_max_q" in step_data and step_data["avg_max_q"] is not None:
            self.avg_max_qs.append(step_data["avg_max_q"])
        if "beta" in step_data and step_data["beta"] is not None:
            self.current_beta = step_data["beta"]
            self.beta_values.append(self.current_beta)
        if "buffer_size" in step_data and step_data["buffer_size"] is not None:
            self.current_buffer_size = step_data["buffer_size"]
            self.buffer_sizes.append(self.current_buffer_size)
        if "lr" in step_data and step_data["lr"] is not None:
            self.current_lr = step_data["lr"]
            self.lr_values.append(self.current_lr)
        if "epsilon" in step_data and step_data["epsilon"] is not None:
            self.current_epsilon = step_data["epsilon"]
            self.epsilon_values.append(self.current_epsilon)

        if "step_time" in step_data and step_data["step_time"] > 1e-9:
            num_steps = step_data.get("num_steps_processed", 1)
            sps = num_steps / step_data["step_time"]
            self.sps_values.append(sps)
            self.current_sps = sps

        # Record resource usage
        if "cpu_usage" in step_data and step_data["cpu_usage"] is not None:
            self.cpu_usage.append(step_data["cpu_usage"])
            self.current_cpu_usage = step_data["cpu_usage"]
        if "memory_usage" in step_data and step_data["memory_usage"] is not None:
            self.memory_usage.append(step_data["memory_usage"])
            self.current_memory_usage = step_data["memory_usage"]
        # Changed key for GPU memory
        if (
            "gpu_memory_usage_percent" in step_data
            and step_data["gpu_memory_usage_percent"] is not None
        ):
            self.gpu_memory_usage_percent.append(step_data["gpu_memory_usage_percent"])
            self.current_gpu_memory_usage_percent = step_data[
                "gpu_memory_usage_percent"
            ]

        return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        if current_global_step is None:
            current_global_step = self.current_global_step

        summary_window = self.summary_avg_window

        def safe_mean(q: Deque, default=0.0) -> float:
            window_data = list(q)[-summary_window:]
            finite_data = [x for x in window_data if np.isfinite(x)]
            return float(np.mean(finite_data)) if finite_data else default

        summary = {
            "avg_score_window": safe_mean(self.episode_scores),
            "avg_length_window": safe_mean(self.episode_lengths),
            "policy_loss": safe_mean(self.policy_losses),
            "value_loss": safe_mean(self.value_losses),
            "entropy": safe_mean(self.entropies),
            "avg_max_q_window": safe_mean(self.avg_max_qs),
            "avg_game_score_window": safe_mean(self.game_scores),
            "avg_triangles_cleared_window": safe_mean(self.episode_triangles_cleared),
            "avg_sps_window": safe_mean(self.sps_values, default=self.current_sps),
            "avg_lr_window": safe_mean(self.lr_values, default=self.current_lr),
            "total_episodes": self.total_episodes,
            "beta": self.current_beta,
            "buffer_size": self.current_buffer_size,
            "steps_per_second": self.current_sps,
            "global_step": current_global_step,
            "current_lr": self.current_lr,
            "best_score": self.best_score,
            "previous_best_score": self.previous_best_score,
            "best_score_step": self.best_score_step,
            "best_game_score": self.best_game_score,
            "previous_best_game_score": self.previous_best_game_score,
            "best_game_score_step": self.best_game_score_step,
            "best_loss": self.best_value_loss,
            "previous_best_loss": self.previous_best_value_loss,
            "best_loss_step": self.best_value_loss_step,
            "num_ep_scores": len(self.episode_scores),
            "num_losses": len(self.value_losses),
            "summary_avg_window_size": summary_window,
            "start_time": self.start_time,
            "training_target_step": self.training_target_step,
            # Add current resource usage
            "current_cpu_usage": self.current_cpu_usage,
            "current_memory_usage": self.current_memory_usage,
            "current_gpu_memory_usage_percent": self.current_gpu_memory_usage_percent,  # Changed key
        }
        return summary

    def get_plot_data(self) -> Dict[str, Deque]:
        return {
            "episode_scores": self.episode_scores.copy(),
            "episode_lengths": self.episode_lengths.copy(),
            "policy_loss": self.policy_losses.copy(),
            "value_loss": self.value_losses.copy(),
            "entropy": self.entropies.copy(),
            "avg_max_qs": self.avg_max_qs.copy(),
            "game_scores": self.game_scores.copy(),
            "episode_triangles_cleared": self.episode_triangles_cleared.copy(),
            "sps_values": self.sps_values.copy(),
            "buffer_sizes": self.buffer_sizes.copy(),
            "beta_values": self.beta_values.copy(),
            "best_rl_score_history": self.best_rl_score_history.copy(),
            "best_game_score_history": self.best_game_score_history.copy(),
            "lr_values": self.lr_values.copy(),
            "epsilon_values": self.epsilon_values.copy(),
            # Resource Usage
            "cpu_usage": self.cpu_usage.copy(),
            "memory_usage": self.memory_usage.copy(),
            "gpu_memory_usage_percent": self.gpu_memory_usage_percent.copy(),  # Changed key
        }

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the aggregator for saving."""
        state = {
            # Deques
            "policy_losses": list(self.policy_losses),
            "value_losses": list(self.value_losses),
            "entropies": list(self.entropies),
            "grad_norms": list(self.grad_norms),
            "avg_max_qs": list(self.avg_max_qs),
            "episode_scores": list(self.episode_scores),
            "episode_lengths": list(self.episode_lengths),
            "game_scores": list(self.game_scores),
            "episode_triangles_cleared": list(self.episode_triangles_cleared),
            "sps_values": list(self.sps_values),
            "buffer_sizes": list(self.buffer_sizes),
            "beta_values": list(self.beta_values),
            "best_rl_score_history": list(self.best_rl_score_history),
            "best_game_score_history": list(self.best_game_score_history),
            "lr_values": list(self.lr_values),
            "epsilon_values": list(self.epsilon_values),
            "cpu_usage": list(self.cpu_usage),
            "memory_usage": list(self.memory_usage),
            "gpu_memory_usage_percent": list(
                self.gpu_memory_usage_percent
            ),  # Changed key
            # Scalar State Variables
            "total_episodes": self.total_episodes,
            "total_triangles_cleared": self.total_triangles_cleared,
            "current_epsilon": self.current_epsilon,
            "current_beta": self.current_beta,
            "current_buffer_size": self.current_buffer_size,
            "current_global_step": self.current_global_step,
            "current_sps": self.current_sps,
            "current_lr": self.current_lr,
            "start_time": self.start_time,
            "training_target_step": self.training_target_step,
            "current_cpu_usage": self.current_cpu_usage,
            "current_memory_usage": self.current_memory_usage,
            "current_gpu_memory_usage_percent": self.current_gpu_memory_usage_percent,  # Changed key
            # Best Value Tracking
            "best_score": self.best_score,
            "previous_best_score": self.previous_best_score,
            "best_score_step": self.best_score_step,
            "best_game_score": self.best_game_score,
            "previous_best_game_score": self.previous_best_game_score,
            "best_game_score_step": self.best_game_score_step,
            "best_value_loss": self.best_value_loss,
            "previous_best_value_loss": self.previous_best_value_loss,
            "best_value_loss_step": self.best_value_loss_step,
            # Config
            "plot_window": self.plot_window,
            "avg_windows": self.avg_windows,
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the state of the aggregator from a dictionary."""
        print("[StatsAggregator] Loading state...")
        self.plot_window = state_dict.get("plot_window", self.plot_window)
        deque_keys = [
            "policy_losses",
            "value_losses",
            "entropies",
            "grad_norms",
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "sps_values",
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",  # Changed key
        ]
        for key in deque_keys:
            if key in state_dict:
                try:
                    data = state_dict[key]
                    if isinstance(data, (list, tuple)):
                        setattr(self, key, deque(data, maxlen=self.plot_window))
                    else:
                        print(
                            f"  -> Warning: Invalid type for deque '{key}' in state_dict: {type(data)}. Skipping."
                        )
                except Exception as e:
                    print(f"  -> Error loading deque '{key}': {e}. Resetting.")
                    setattr(self, key, deque(maxlen=self.plot_window))
            else:
                print(
                    f"  -> Warning: Deque '{key}' not found in state_dict. Resetting."
                )
                setattr(self, key, deque(maxlen=self.plot_window))

        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_epsilon",
            "current_beta",
            "current_buffer_size",
            "current_global_step",
            "current_sps",
            "current_lr",
            "start_time",
            "training_target_step",
            "current_cpu_usage",
            "current_memory_usage",
            "current_gpu_memory_usage_percent",  # Changed key
        ]
        default_values = {"start_time": time.time(), "training_target_step": 0}
        for key in scalar_keys:
            if key in state_dict:
                setattr(self, key, state_dict[key])
            else:
                default_val = default_values.get(
                    key, 0.0 if isinstance(getattr(self, key, 0.0), float) else 0
                )
                setattr(self, key, default_val)
                print(
                    f"  -> Warning: Scalar '{key}' not found in state_dict. Using default ({default_val})."
                )

        best_value_keys = [
            "best_score",
            "previous_best_score",
            "best_score_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
        ]
        for key in best_value_keys:
            if key in state_dict:
                setattr(self, key, state_dict[key])
            else:
                print(
                    f"  -> Warning: Best value key '{key}' not found in state_dict. Using default."
                )

        self.avg_windows = state_dict.get("avg_windows", self.avg_windows)
        self.summary_avg_window = self.avg_windows[0] if self.avg_windows else 100

        print("[StatsAggregator] State loaded.")
        print(f"  -> Loaded total_episodes: {self.total_episodes}")
        print(f"  -> Loaded best_score: {self.best_score}")
        print(
            f"  -> Loaded start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}"
        )
        print(f"  -> Loaded training_target_step: {self.training_target_step}")
