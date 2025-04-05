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

        self.policy_losses: Deque[float] = deque(maxlen=plot_window)
        self.value_losses: Deque[float] = deque(maxlen=plot_window)
        self.entropies: Deque[float] = deque(maxlen=plot_window)
        self.grad_norms: Deque[float] = deque(maxlen=plot_window)
        self.avg_max_qs: Deque[float] = deque(maxlen=plot_window)
        self.episode_scores: Deque[float] = deque(maxlen=plot_window)
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_lines_cleared: Deque[int] = deque(maxlen=plot_window)
        self.sps_values: Deque[float] = deque(maxlen=plot_window)
        self.buffer_sizes: Deque[int] = deque(maxlen=plot_window)
        self.beta_values: Deque[float] = deque(maxlen=plot_window)
        self.best_rl_score_history: Deque[float] = deque(maxlen=plot_window)
        self.best_game_score_history: Deque[int] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(maxlen=plot_window)
        self.epsilon_values: Deque[float] = deque(maxlen=plot_window)

        self.total_episodes = 0
        self.total_lines_cleared = 0
        self.current_epsilon: float = 0.0
        self.current_beta: float = 0.0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0
        self.current_sps: float = 0.0
        self.current_lr: float = 0.0

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
        lines_cleared: Optional[int] = None,
    ) -> Dict[str, Any]:
        current_step = (
            global_step if global_step is not None else self.current_global_step
        )
        update_info = {"new_best_rl": False, "new_best_game": False}

        # --- DEBUG LOGGING ---
        # print(f"[StatsAggregator Debug] Appending Ep {episode_num}: RLScore={episode_score:.2f}, Len={episode_length}, GameScore={game_score}")
        # --- END DEBUG LOGGING ---

        self.episode_scores.append(episode_score)
        self.episode_lengths.append(episode_length)
        if game_score is not None:
            self.game_scores.append(game_score)
        if lines_cleared is not None:
            self.episode_lines_cleared.append(lines_cleared)
            self.total_lines_cleared += lines_cleared
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

        update_info = {"new_best_loss": False}

        if "policy_loss" in step_data and step_data["policy_loss"] is not None:
            # --- DEBUG LOGGING ---
            # print(f"[StatsAggregator Debug] Appending Policy Loss: {step_data['policy_loss']:.4f} at Step {g_step}")
            # --- END DEBUG LOGGING ---
            self.policy_losses.append(step_data["policy_loss"])

        if "value_loss" in step_data and step_data["value_loss"] is not None:
            current_value_loss = step_data["value_loss"]
            # --- DEBUG LOGGING ---
            # print(f"[StatsAggregator Debug] Appending Value Loss: {current_value_loss:.4f} at Step {g_step}")
            # --- END DEBUG LOGGING ---
            self.value_losses.append(current_value_loss)
            if current_value_loss < self.best_value_loss and g_step > 0:
                self.previous_best_value_loss = self.best_value_loss
                self.best_value_loss = current_value_loss
                self.best_value_loss_step = g_step
                update_info["new_best_loss"] = True

        if "entropy" in step_data and step_data["entropy"] is not None:
            # --- DEBUG LOGGING ---
            # print(f"[StatsAggregator Debug] Appending Entropy: {step_data['entropy']:.4f} at Step {g_step}")
            # --- END DEBUG LOGGING ---
            self.entropies.append(step_data["entropy"])

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

        return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        if current_global_step is None:
            current_global_step = self.current_global_step

        summary_window = self.summary_avg_window

        def safe_mean(q: Deque, default=0.0) -> float:
            window_data = list(q)[-summary_window:]
            return float(np.mean(window_data)) if window_data else default

        summary = {
            "avg_score_window": safe_mean(self.episode_scores),
            "avg_length_window": safe_mean(self.episode_lengths),
            "policy_loss": safe_mean(self.policy_losses),
            "value_loss": safe_mean(self.value_losses),
            "entropy": safe_mean(self.entropies),
            "avg_max_q_window": safe_mean(self.avg_max_qs),
            "avg_game_score_window": safe_mean(self.game_scores),
            "avg_lines_cleared_window": safe_mean(self.episode_lines_cleared),
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
        }
        return summary

    def get_plot_data(self) -> Dict[str, Deque]:
        # --- DEBUG LOGGING ---
        # print(f"[StatsAggregator Debug] get_plot_data lengths: "
        #       f"RLScore={len(self.episode_scores)}, GameScore={len(self.game_scores)}, EpLen={len(self.episode_lengths)}, "
        #       f"PLoss={len(self.policy_losses)}, VLoss={len(self.value_losses)}, Ent={len(self.entropies)}")
        # --- END DEBUG LOGGING ---
        return {
            "episode_scores": self.episode_scores.copy(),
            "episode_lengths": self.episode_lengths.copy(),
            "policy_loss": self.policy_losses.copy(),
            "value_loss": self.value_losses.copy(),
            "entropy": self.entropies.copy(),
            "avg_max_qs": self.avg_max_qs.copy(),
            "game_scores": self.game_scores.copy(),
            "episode_lines_cleared": self.episode_lines_cleared.copy(),
            "sps_values": self.sps_values.copy(),
            "buffer_sizes": self.buffer_sizes.copy(),
            "beta_values": self.beta_values.copy(),
            "best_rl_score_history": self.best_rl_score_history.copy(),
            "best_game_score_history": self.best_game_score_history.copy(),
            "lr_values": self.lr_values.copy(),
            "epsilon_values": self.epsilon_values.copy(),
        }
