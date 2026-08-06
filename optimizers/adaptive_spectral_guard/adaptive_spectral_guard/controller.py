"""Adaptive WeightWatcher-driven per-layer controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import math
import numpy as np
import pandas as pd

from .config import GuardConfig, LayerPolicy


@dataclass
class LayerState:
    parameter: str
    epoch: int = -1

    regime: str = "off"
    safe_epochs: int = 0

    alpha: float = math.nan
    previous_alpha: float = math.nan
    alpha_trend: float = math.nan

    erg_gap: float = math.nan
    detX_num: int = 0
    num_pl_spikes: int = 0
    midpoint_rank: int = 0
    previous_midpoint_rank: int = 0

    boundary_overlap_ratio: float = 0.0
    support_change_ratio: float = math.nan
    erg_gap_ratio: float = math.nan

    # ``confidence`` remains the compatibility name and equals the smoothed
    # confidence in the stabilized controller.
    raw_confidence: float = math.nan
    smoothed_confidence: float = math.nan
    confidence: float = 0.0
    volume_confidence: float = 0.0
    shape_confidence: float = 0.0

    beta_E: float = math.nan
    beta_reliable: bool = False

    task_conflict_ema: float = 0.0
    task_harmful_fraction: float = 0.0
    task_throttle: float = 1.0

    base_gain: float = 0.0
    volume_effective_gain: float = 0.0
    shape_effective_gain: float = 0.0
    effective_gain: float = 0.0
    shape_active: bool = False

    reason: str = "uninitialized"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveSpectralController:
    """Slow controller refreshed from WeightWatcher once per epoch."""

    def __init__(self, config: GuardConfig) -> None:
        config.validate()
        self.config = config
        self.states: dict[str, LayerState] = {}

    @staticmethod
    def _safe_float(value: Any, default: float = math.nan) -> float:
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if value is None or pd.isna(value):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _resolve_parameter(self, supplied: str) -> Optional[str]:
        if supplied in self.config.policies:
            return supplied
        candidates = [supplied]
        if not supplied.endswith(".weight"):
            candidates.append(f"{supplied}.weight")
        for candidate in candidates:
            if candidate in self.config.policies:
                return candidate
        matches = [
            key
            for key in self.config.policies
            if any(
                key.endswith(candidate) or candidate.endswith(key)
                for candidate in candidates
            )
        ]
        return matches[0] if len(matches) == 1 else None

    def _raw_confidence(
        self,
        *,
        overlap: float,
        support_change: float,
        erg_gap_ratio: float,
    ) -> float:
        c = self.config.controller
        if not math.isfinite(support_change):
            support_factor = 1.0
        else:
            support_factor = math.exp(
                -max(support_change, 0.0) / c.support_change_scale
            )
        gap_factor = math.exp(
            -max(erg_gap_ratio, 0.0) / c.erg_gap_ratio_scale
        )
        return float(np.clip(overlap * support_factor * gap_factor, 0.0, 1.0))

    def _smooth_confidence(
        self,
        raw: float,
        previous: LayerState,
    ) -> float:
        decay = self.config.controller.confidence_ema_decay
        previous_value = previous.smoothed_confidence
        if not math.isfinite(previous_value) and previous.epoch >= 0:
            previous_value = previous.confidence
        if not math.isfinite(previous_value):
            return float(raw)
        return float(
            np.clip(
                decay * previous_value + (1.0 - decay) * raw,
                0.0,
                1.0,
            )
        )

    def _transition(
        self,
        state: LayerState,
        policy: LayerPolicy,
    ) -> tuple[str, int, str]:
        c = self.config.controller
        if not policy.enabled:
            return "off", 0, "policy disabled"
        if not math.isfinite(state.alpha):
            return "off", 0, "missing alpha"

        # The original controller remains the default. V2 applies confidence
        # separately to the volume and shape channels instead of turning the
        # entire layer off.
        if (
            not c.separate_channel_confidence
            and state.confidence < c.min_confidence
        ):
            return "off", 0, "low ECS confidence"

        falling_fast = (
            math.isfinite(state.alpha_trend)
            and state.alpha_trend <= c.alpha_trend_on
            and state.alpha <= c.trend_ceiling
        )
        strong = state.alpha <= c.alpha_strong
        weak = state.alpha <= c.alpha_on or falling_fast

        if strong:
            return "strong", 0, "alpha below strong boundary"
        if weak:
            return "weak", 0, (
                "alpha near boundary"
                if state.alpha <= c.alpha_on
                else "alpha falling rapidly"
            )

        if state.regime in {"weak", "strong"}:
            safe_now = (
                state.alpha >= c.alpha_off
                and (
                    not math.isfinite(state.alpha_trend)
                    or state.alpha_trend >= -0.005
                )
            )
            safe_epochs = state.safe_epochs + 1 if safe_now else 0
            if safe_epochs < c.off_patience:
                return "weak", safe_epochs, "hysteresis hold"
            return "off", safe_epochs, "safe for off-patience window"

        return "off", 0, "alpha safely above boundary"

    def _channel_confidences(
        self,
        state: LayerState,
    ) -> tuple[float, float]:
        c = self.config.controller
        if not c.separate_channel_confidence:
            value = float(np.clip(state.confidence, 0.0, 1.0))
            return value, value

        volume = float(np.clip(state.smoothed_confidence, 0.0, 1.0))
        if (
            state.regime != "off"
            and state.alpha <= c.volume_confidence_floor_alpha
        ):
            volume = max(
                volume,
                c.volume_confidence_floor_below_boundary,
            )

        shape = float(np.clip(state.smoothed_confidence, 0.0, 1.0))
        if (
            shape < c.shape_min_confidence
            or state.raw_confidence < c.shape_raw_confidence_floor
        ):
            shape = 0.0
        return volume, shape

    def _refresh_effective_gains(
        self,
        state: LayerState,
    ) -> None:
        state.volume_effective_gain = (
            state.base_gain
            * state.volume_confidence
            * state.task_throttle
        )
        state.shape_effective_gain = (
            state.base_gain
            * state.shape_confidence
            * state.task_throttle
        )
        state.effective_gain = max(
            state.volume_effective_gain,
            state.shape_effective_gain,
        )

    def update_from_weightwatcher(
        self,
        metrics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Update states from successful WeightWatcher rows.

        The input must contain direct WeightWatcher alpha and ERG_gap values.
        """

        required = {
            "parameter_name",
            "epoch",
            "status",
            "alpha",
            "alpha_source",
            "ERG_gap",
            "ERG_gap_source",
            "detX_num",
            "num_pl_spikes",
            "m_midpoint",
            "boundary_overlap_ratio",
        }
        missing = required - set(metrics.columns)
        if missing:
            raise ValueError(f"Missing WeightWatcher columns: {sorted(missing)}")

        successful = metrics.loc[metrics["status"].eq("ok")].copy()
        if not successful.empty:
            if not successful["alpha_source"].astype(str).eq(
                "WeightWatcher"
            ).all():
                raise ValueError("alpha must come directly from WeightWatcher")
            if not successful["ERG_gap_source"].astype(str).eq(
                "WeightWatcher"
            ).all():
                raise ValueError("ERG_gap must come directly from WeightWatcher")

        updated: list[dict[str, Any]] = []
        for _, row in successful.iterrows():
            supplied = str(row["parameter_name"])
            parameter = self._resolve_parameter(supplied)
            if parameter is None:
                continue
            policy = self.config.policy_for(parameter)
            previous = self.states.get(parameter, LayerState(parameter=parameter))

            alpha = self._safe_float(row["alpha"])
            midpoint = self._safe_int(row["m_midpoint"])
            previous_midpoint = previous.midpoint_rank
            support_change = (
                abs(midpoint - previous_midpoint) / max(previous_midpoint, 1)
                if previous_midpoint > 0
                else math.nan
            )
            detx = self._safe_int(row["detX_num"])
            mpl = self._safe_int(row["num_pl_spikes"])
            erg_gap = self._safe_float(row["ERG_gap"])
            gap_ratio = abs(erg_gap) / max(detx, mpl, 1)
            overlap = float(
                np.clip(
                    self._safe_float(row["boundary_overlap_ratio"], 0.0),
                    0.0,
                    1.0,
                )
            )
            raw_confidence = self._raw_confidence(
                overlap=overlap,
                support_change=support_change,
                erg_gap_ratio=gap_ratio,
            )
            smoothed_confidence = self._smooth_confidence(
                raw_confidence,
                previous,
            )

            state = LayerState(
                parameter=parameter,
                epoch=self._safe_int(row["epoch"], previous.epoch),
                regime=previous.regime,
                safe_epochs=previous.safe_epochs,
                alpha=alpha,
                previous_alpha=previous.alpha,
                alpha_trend=(
                    alpha - previous.alpha
                    if math.isfinite(alpha) and math.isfinite(previous.alpha)
                    else math.nan
                ),
                erg_gap=erg_gap,
                detX_num=detx,
                num_pl_spikes=mpl,
                midpoint_rank=midpoint,
                previous_midpoint_rank=previous_midpoint,
                boundary_overlap_ratio=overlap,
                support_change_ratio=support_change,
                erg_gap_ratio=gap_ratio,
                raw_confidence=raw_confidence,
                smoothed_confidence=smoothed_confidence,
                confidence=smoothed_confidence,
                beta_E=self._safe_float(
                    row.get("beta_E_midpoint", math.nan)
                ),
                beta_reliable=bool(
                    row.get("scale_balance_reliable", False)
                ),
                task_conflict_ema=previous.task_conflict_ema,
                task_harmful_fraction=previous.task_harmful_fraction,
                task_throttle=previous.task_throttle,
            )

            regime, safe_epochs, reason = self._transition(state, policy)
            state.regime = regime
            state.safe_epochs = safe_epochs
            state.reason = reason

            if regime == "strong":
                state.base_gain = policy.strong_gain
            elif regime == "weak":
                state.base_gain = policy.weak_gain
            else:
                state.base_gain = 0.0

            (
                state.volume_confidence,
                state.shape_confidence,
            ) = self._channel_confidences(state)
            self._refresh_effective_gains(state)

            c = self.config.controller
            beta_trigger = bool(
                state.beta_reliable
                and math.isfinite(state.beta_E)
                and state.beta_E >= c.beta_on
            )
            alpha_trigger = bool(state.alpha <= c.shape_alpha_on)
            if c.shape_requires_alpha_boundary:
                shape_trigger = beta_trigger and alpha_trigger
            else:
                shape_trigger = beta_trigger or alpha_trigger
            state.shape_active = bool(
                regime != "off"
                and state.shape_confidence > 0.0
                and shape_trigger
            )

            self.states[parameter] = state
            updated.append(state.to_dict())

        return pd.DataFrame(updated)

    def observe_task_feedback(self, stats: pd.DataFrame) -> pd.DataFrame:
        """Update per-layer throttles from attempted task conflicts."""

        if stats is None or stats.empty:
            return self.frame()
        if "parameter" not in stats:
            raise ValueError("step statistics are missing parameter")

        c = self.config.controller
        rows = []
        for supplied, group in stats.groupby("parameter"):
            parameter = self._resolve_parameter(str(supplied))
            if parameter is None or parameter not in self.states:
                continue
            state = self.states[parameter]
            attempted = pd.to_numeric(
                group.get("task_conflict_ratio_pre", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            if attempted.empty:
                mean_positive = 0.0
                harmful_fraction = 0.0
            else:
                positive = attempted.clip(lower=0.0)
                mean_positive = float(positive.mean())
                harmful_fraction = float((attempted > 0.0).mean())

            ema = (
                c.task_conflict_ema_decay * state.task_conflict_ema
                + (1.0 - c.task_conflict_ema_decay) * mean_positive
            )
            throttle = max(
                c.minimum_task_throttle,
                math.exp(-c.task_conflict_penalty * ema),
            )
            state.task_conflict_ema = ema
            state.task_harmful_fraction = harmful_fraction
            state.task_throttle = throttle
            self._refresh_effective_gains(state)
            rows.append(state.to_dict())
        return pd.DataFrame(rows)

    def get_state(self, parameter_name: str) -> LayerState:
        parameter = self._resolve_parameter(parameter_name)
        if parameter is None:
            return LayerState(parameter=parameter_name, reason="no policy")
        return self.states.get(
            parameter,
            LayerState(parameter=parameter, reason="awaiting WeightWatcher"),
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [self.states[name].to_dict() for name in sorted(self.states)]
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "states": {
                name: state.to_dict() for name, state in self.states.items()
            }
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.states = {
            name: LayerState(**values)
            for name, values in payload.get("states", {}).items()
        }
