from __future__ import annotations

from smart_warehouse.entities import Task
from smart_warehouse.task_utils import clone_task


def test_clone_task_resets_mutable_execution_state() -> None:
    task = Task(
        task_id="FW-1",
        created_minute=10,
        zone_id="C",
        position=(2, 12),
        assigned_robot_id="R-01",
    )
    clone = clone_task(task)
    assert clone is not task
    assert clone.assigned_robot_id is None
    assert clone.position == task.position
    assert clone.zone_id == task.zone_id
