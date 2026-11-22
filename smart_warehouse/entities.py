from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

Position = tuple[int, int]


class TaskStatus(str, Enum):
    WAITING = "WAITING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class RobotStatus(str, Enum):
    IDLE = "IDLE"
    REPOSITIONING = "REPOSITIONING"
    MOVING_TO_TASK = "MOVING_TO_TASK"
    WORKING = "WORKING"


@dataclass
class Task:
    task_id: str
    created_minute: float
    zone_id: str
    position: Position
    priority: int = 1
    quantity: float = 1.0
    service_ticks: int | None = None
    wave_id: str | None = None
    source_location: str | None = None
    source_sequence: int | None = None
    status: TaskStatus = TaskStatus.WAITING
    assigned_robot_id: Optional[str] = None
    assigned_minute: Optional[float] = None
    started_minute: Optional[float] = None
    completed_minute: Optional[float] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        data["position"] = list(self.position)
        return data


@dataclass
class Robot:
    robot_id: str
    position: Position
    battery: float = 100.0
    status: RobotStatus = RobotStatus.IDLE
    path: list[Position] = field(default_factory=list)
    current_task_id: Optional[str] = None
    target_zone_id: Optional[str] = None
    work_remaining_ticks: int = 0
    total_distance: int = 0
    task_distance: int = 0
    reposition_distance: int = 0
    completed_task_count: int = 0
    idle_action: str = "STAY"
    idle_action_reason: str = ""
    idle_move_started_tick: int | None = None
    idle_move_origin: Position | None = None
    idle_move_purpose: str | None = None

    @property
    def available(self) -> bool:
        return self.status in {RobotStatus.IDLE, RobotStatus.REPOSITIONING}

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        data["position"] = list(self.position)
        data["path"] = [list(point) for point in self.path]
        return data


@dataclass(frozen=True)
class Zone:
    zone_id: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    waiting_position: Position

    def contains(self, position: Position) -> bool:
        x, y = position
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def to_dict(self) -> dict:
        data = asdict(self)
        data["waiting_position"] = list(self.waiting_position)
        return data
