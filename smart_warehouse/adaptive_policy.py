from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean, pstdev

from smart_warehouse.config import SimulationConfig
from smart_warehouse.entities import Robot, RobotStatus


@dataclass(frozen=True)
class PolicyDecision:
    active_policy: str
    score: float
    reason: str
    top_zone: str
    signals: dict[str, float | int | str | bool]
    changed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class AdaptivePolicyController:
    """Select REACTIVE or PREDICTIVE behavior using operational thresholds.

    Allocation itself is always reactive: released tasks are assigned to an
    available robot. The controller only decides whether idle robots may be
    proactively repositioned before future work arrives.

    Hard safety gates always select REACTIVE behavior:
      * the queue is already overloaded;
      * too few robots are idle to reposition safely;
      * the forecast has too little demand or no spatial concentration.

    When gates pass, a 0..1 suitability score combines spatial concentration,
    separation between the two busiest zones, idle capacity, and recent demand
    stability. Separate enter/exit thresholds add hysteresis so the controller
    does not flicker between policies.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.active_policy = "REACTIVE"
        self.last_switch_tick = -10**9
        self.last_decision = PolicyDecision(
            active_policy="REACTIVE",
            score=0.0,
            reason="Waiting for enough demand history before proactive movement.",
            top_zone="A",
            signals={},
            changed=False,
        )

    @staticmethod
    def _stability_score(history: list[int]) -> float:
        if len(history) < 3:
            return 0.50
        values = [float(value) for value in history[-6:]]
        mean = fmean(values)
        if mean <= 1e-9:
            return 0.50
        coefficient_of_variation = pstdev(values) / mean
        return _clip(1.0 / (1.0 + coefficient_of_variation))

    def evaluate(
        self,
        *,
        predictions: dict[str, float],
        queue_counts: dict[str, int],
        robots: list[Robot],
        recent_counts: dict[str, list[int]],
        tick: int,
        forecast_notice_active: bool = False,
    ) -> PolicyDecision:
        zones = sorted(predictions)
        if not zones:
            raise ValueError("predictions must contain at least one zone")

        ordered = sorted(zones, key=lambda zone: (float(predictions[zone]), zone), reverse=True)
        top_zone = ordered[0]
        second_zone = ordered[1] if len(ordered) > 1 else top_zone
        total_forecast = sum(max(0.0, float(value)) for value in predictions.values())
        top_forecast = max(0.0, float(predictions[top_zone]))
        second_forecast = max(0.0, float(predictions[second_zone]))
        top_share = top_forecast / total_forecast if total_forecast > 1e-9 else 0.0
        forecast_gap = max(0.0, top_forecast - second_forecast)

        robot_count = max(1, len(robots))
        idle_count = sum(robot.status == RobotStatus.IDLE for robot in robots)
        idle_ratio = idle_count / robot_count
        waiting_total = int(sum(max(0, int(value)) for value in queue_counts.values()))
        queue_per_robot = waiting_total / robot_count
        stability = self._stability_score(recent_counts.get(top_zone, []))

        concentration_score = _clip((top_share - 0.25) / 0.50)
        average_zone_forecast = total_forecast / max(1, len(zones))
        gap_score = _clip(forecast_gap / max(1.0, average_zone_forecast))
        capacity_score = _clip(
            (idle_ratio - self.config.adaptive_min_idle_ratio)
            / max(1e-9, 1.0 - self.config.adaptive_min_idle_ratio)
        )
        notice_bonus = 0.12 if forecast_notice_active else 0.0
        score = _clip(
            0.35 * concentration_score
            + 0.25 * gap_score
            + 0.20 * capacity_score
            + 0.20 * stability
            + notice_bonus
        )

        signals: dict[str, float | int | str | bool] = {
            "top_zone": top_zone,
            "top_forecast": round(top_forecast, 3),
            "total_forecast": round(total_forecast, 3),
            "top_zone_share": round(top_share, 3),
            "forecast_gap": round(forecast_gap, 3),
            "idle_robots": idle_count,
            "idle_ratio": round(idle_ratio, 3),
            "waiting_tasks": waiting_total,
            "queue_per_robot": round(queue_per_robot, 3),
            "recent_stability": round(stability, 3),
            "forecast_notice_active": forecast_notice_active,
            "enter_threshold": self.config.adaptive_enter_score,
            "regular_enter_threshold": self.config.adaptive_regular_enter_score,
            "exit_threshold": self.config.adaptive_exit_score,
        }

        hard_reason: str | None = None
        if queue_per_robot >= self.config.adaptive_overload_queue_per_robot:
            hard_reason = (
                f"REACTIVE: backlog is already high ({waiting_total} waiting, "
                f"{queue_per_robot:.1f} per robot), so extra repositioning would waste capacity."
            )
        elif idle_ratio < self.config.adaptive_min_idle_ratio:
            hard_reason = (
                f"REACTIVE: only {idle_count}/{robot_count} robots are idle; current tasks take priority."
            )
        elif total_forecast < self.config.adaptive_min_forecast_total:
            hard_reason = (
                f"REACTIVE: forecast volume is too small ({total_forecast:.1f}) to justify proactive travel."
            )
        elif (
            not forecast_notice_active
            and top_forecast < self.config.adaptive_regular_min_top_forecast
        ):
            hard_reason = (
                f"REACTIVE: the regular forecast hotspot is only {top_forecast:.1f} tasks; "
                f"at least {self.config.adaptive_regular_min_top_forecast:.1f} is required without an advance notice."
            )
        elif top_share < self.config.adaptive_min_top_zone_share:
            hard_reason = (
                f"REACTIVE: demand is spread across zones; the largest share is only {top_share:.0%}."
            )
        elif forecast_gap < self.config.adaptive_min_forecast_gap:
            hard_reason = (
                f"REACTIVE: the top two zones differ by only {forecast_gap:.1f} tasks."
            )

        previous = self.active_policy
        can_switch = tick - self.last_switch_tick >= self.config.adaptive_min_policy_hold_ticks
        required_enter_score = (
            self.config.adaptive_enter_score
            if forecast_notice_active
            else self.config.adaptive_regular_enter_score
        )
        signals["required_enter_threshold"] = required_enter_score

        if hard_reason is not None:
            # Hard safety gates are allowed to leave predictive mode immediately.
            self.active_policy = "REACTIVE"
            reason = hard_reason
        elif self.active_policy == "PREDICTIVE":
            if score < self.config.adaptive_exit_score and can_switch:
                self.active_policy = "REACTIVE"
                reason = (
                    f"REACTIVE: predictive suitability fell to {score:.2f}, below the "
                    f"exit threshold {self.config.adaptive_exit_score:.2f}."
                )
            else:
                reason = (
                    f"PREDICTIVE: Zone {top_zone} has {top_share:.0%} of forecast demand and "
                    f"{idle_count} robots are available for safe repositioning."
                )
        else:
            if score >= required_enter_score and can_switch:
                self.active_policy = "PREDICTIVE"
                signal_name = "advance-notice" if forecast_notice_active else "regular forecast"
                reason = (
                    f"PREDICTIVE: {signal_name} suitability {score:.2f} exceeds the required "
                    f"threshold {required_enter_score:.2f}; Zone {top_zone} is the clear hotspot."
                )
            elif score >= required_enter_score and not can_switch:
                reason = (
                    f"REACTIVE: predictive signal is strong ({score:.2f}), but the "
                    "minimum hold time prevents unstable rapid switching."
                )
            else:
                reason = (
                    f"REACTIVE: predictive suitability is {score:.2f}, below the required "
                    f"threshold {required_enter_score:.2f}."
                )

        changed = self.active_policy != previous
        if changed:
            self.last_switch_tick = tick

        self.last_decision = PolicyDecision(
            active_policy=self.active_policy,
            score=round(score, 3),
            reason=reason,
            top_zone=top_zone,
            signals=signals,
            changed=changed,
        )
        return self.last_decision
