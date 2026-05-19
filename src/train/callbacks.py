"""src/train/callbacks.py — DetailedEvalCallback for Phase 3

Extends MaskableEvalCallback (sb3_contrib) to:
  1. Sync VecNormalize stats from train env to eval env
  2. Log EENS, diesel, stockouts, emergency_arrivals vs training steps
     to both TensorBoard and MLflow at each evaluation checkpoint.
  3. Pass action masks during MaskablePPO evaluation (critical — without
     masks, best-model selection is based on unmasked actions).

This produces the training curves the reviewer asked for:
  reward vs training steps       (SB3 default)
  EENS vs training steps         (NEW — reliability learning)
  diesel vs training steps       (NEW — cost learning)
  stockouts vs training steps    (NEW — safety learning)
  emergency_arrivals vs steps    (NEW — supply chain metric)
  reward/episode_mean/std/min/max (NEW — reward magnitude monitoring)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as _BaseEvalCallback
    from sb3_contrib.common.maskable.utils import get_action_masks
    _MASKABLE_AVAILABLE = True
except ImportError:
    # Fallback if sb3_contrib version doesn't have MaskableEvalCallback
    from stable_baselines3.common.callbacks import EvalCallback as _BaseEvalCallback
    _MASKABLE_AVAILABLE = False

from stable_baselines3.common.vec_env import VecNormalize


class SyncNormEvalCallback(_BaseEvalCallback):
    """Drop-in EvalCallback that syncs VecNormalize stats before each eval pass."""

    def _sync_vecnormalize(self):
        train_vn = self.model.get_vec_normalize_env()
        eval_vn  = self.eval_env
        if not isinstance(train_vn, VecNormalize):
            return
        if not isinstance(eval_vn, VecNormalize):
            return
        eval_vn.obs_rms    = train_vn.obs_rms
        eval_vn.ret_rms    = train_vn.ret_rms
        eval_vn.training   = False
        eval_vn.norm_reward = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            self._sync_vecnormalize()
        return super()._on_step()


class DetailedEvalCallback(_BaseEvalCallback):
    """EvalCallback that syncs VecNormalize AND logs operational metrics per checkpoint.

    Inherits from MaskableEvalCallback when available — ensures best-model
    selection uses action masks (critical for MaskablePPO correctness).

    Additional metrics logged at each eval checkpoint (TensorBoard + MLflow):
      eval/EENS_kWh          — expected energy not served per episode
      eval/diesel_kWh        — diesel consumed per episode
      eval/stockout_events   — stockout events per episode
      eval/emergency_arrivals— deliveries that arrived after a stockout
      eval/uptime_pct        — fraction of timesteps with load fully served
      eval/mean_inv_pct      — mean inventory level as % of tank capacity
      reward/episode_mean    — mean episode total reward (reward magnitude monitor)
      reward/episode_std     — std of episode total reward
      reward/episode_min/max — range of episode total reward
    """

    def __init__(
        self,
        *args,
        site: str = "",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._site       = site
        self._seed       = seed
        self._step_log: List[Dict[str, Any]] = []

    def _sync_vecnormalize(self):
        """Sync VecNormalize stats from training env to eval env."""
        train_vn = self.model.get_vec_normalize_env()
        eval_vn  = self.eval_env
        if not isinstance(train_vn, VecNormalize):
            return
        if not isinstance(eval_vn, VecNormalize):
            return
        eval_vn.obs_rms    = train_vn.obs_rms
        eval_vn.ret_rms    = train_vn.ret_rms
        eval_vn.training   = False
        eval_vn.norm_reward = False

    def _on_step(self) -> bool:
        is_eval_step = (
            self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0)
        )
        if is_eval_step:
            self._sync_vecnormalize()

        result = super()._on_step()

        if is_eval_step:
            self._log_detailed_metrics()

        return result

    def _log_detailed_metrics(self):
        try:
            self._collect_and_log_metrics()
        except Exception:
            pass  # never let metric logging crash training

    def _collect_and_log_metrics(self):
        """Run a short evaluation pass with action masks to collect episode_stats."""
        n_target = self.n_eval_episodes
        all_stats: List[Dict] = []

        obs = self.eval_env.reset()
        n_done = 0

        while n_done < n_target:
            # ── Pass action masks for MaskablePPO ─────────────────────────────
            if _MASKABLE_AVAILABLE:
                try:
                    action_masks = get_action_masks(self.eval_env)
                    action, _ = self.model.predict(
                        obs, action_masks=action_masks,
                        deterministic=self.deterministic
                    )
                except (AttributeError, TypeError):
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
            else:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)

            obs, _, done, info = self.eval_env.step(action)

            for i, d in enumerate(done):
                if d:
                    ep_info = info[i] if isinstance(info, list) else info
                    stats = ep_info.get("episode_stats", ep_info)
                    if isinstance(stats, dict) and "EENS_kWh" in stats:
                        all_stats.append(stats)
                    n_done += 1
                    if n_done >= n_target:
                        break

        if not all_stats:
            return

        def avg(key, default=0.0):
            vals = [s.get(key, default) for s in all_stats]
            return float(np.mean(vals)) if vals else default

        step = self.num_timesteps
        metrics = {
            "eval/EENS_kWh":           avg("EENS_kWh"),
            "eval/diesel_kWh":         avg("diesel_kWh"),
            "eval/stockout_events":    avg("stockout_events"),
            "eval/emergency_arrivals": avg("emergency_arrivals", 0),
            "eval/uptime_pct":         avg("uptime_pct"),
            "eval/mean_inv_pct":       avg("mean_inv_pct"),
        }

        # ── Reward magnitude stats — inspect before deciding on norm_reward ───
        ep_rewards = [s.get("total_reward", np.nan) for s in all_stats
                      if "total_reward" in s]
        if ep_rewards and not all(np.isnan(ep_rewards)):
            r = np.array([x for x in ep_rewards if not np.isnan(x)])
            metrics.update({
                "reward/episode_mean": float(np.mean(r)),
                "reward/episode_std":  float(np.std(r)),
                "reward/episode_min":  float(np.min(r)),
                "reward/episode_max":  float(np.max(r)),
            })

        # Log to TensorBoard
        for k, v in metrics.items():
            self.logger.record(k, v)

        # Log to MLflow if a run is active
        if _MLFLOW_AVAILABLE:
            try:
                if mlflow.active_run() is not None:
                    mlflow.log_metrics(
                        {k.replace("/", "_"): v for k, v in metrics.items()},
                        step=step,
                    )
            except Exception:
                pass

        self._step_log.append({"timestep": step, **metrics})

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Return the full logged history for generating training curve plots."""
        return self._step_log