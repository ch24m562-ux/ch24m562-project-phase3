"""
mlflow_rollout_callback.py

Small additive callback that logs SB3's rollout-level episode reward
(computed directly from model.ep_info_buffer, the same buffer SB3 uses
internally for rollout/ep_rew_mean) to MLflow during training.
Does NOT modify any existing training logic -- purely additive and
read-only with respect to the model, environment, and training loop.

Usage in train_rl_inv.py:
    from mlflow_rollout_callback import MlflowRolloutCallback
    ...
    rollout_cb = MlflowRolloutCallback()
    model.learn(total_timesteps=args.timesteps,
                callback=[eval_cb, rollout_cb])   # combine with existing callback
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import mlflow


class MlflowRolloutCallback(BaseCallback):
    """
    Logs the mean episode reward and length, computed directly from
    self.model.ep_info_buffer (the deque SB3 populates from each
    Monitor-wrapped env's episode 'info' dict on episode completion).

    Reading ep_info_buffer directly avoids relying on the timing of
    SB3's own internal logger.dump() call, which is not guaranteed to
    align with _on_rollout_end(). This callback only READS the buffer;
    it never writes to it, the model, or the environment.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = getattr(self.model, "ep_info_buffer", None)
        if not buf or len(buf) == 0:
            return  # no completed episodes yet -- nothing to log

        ep_rewards = [ep["r"] for ep in buf if "r" in ep]
        ep_lengths = [ep["l"] for ep in buf if "l" in ep]

        metrics = {}
        if ep_rewards:
            metrics["rollout/ep_rew_mean"] = float(np.mean(ep_rewards))
        if ep_lengths:
            metrics["rollout/ep_len_mean"] = float(np.mean(ep_lengths))

        if metrics:
            mlflow.log_metrics(metrics, step=self.num_timesteps)
