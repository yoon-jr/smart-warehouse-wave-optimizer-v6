from __future__ import annotations

from dataclasses import replace

from smart_warehouse.entities import Task, TaskStatus


def clone_task(task: Task) -> Task:
    """Return an execution-independent copy of a source task."""
    return replace(
        task,
        status=TaskStatus.WAITING,
        assigned_robot_id=None,
        assigned_minute=None,
        started_minute=None,
        completed_minute=None,
    )
