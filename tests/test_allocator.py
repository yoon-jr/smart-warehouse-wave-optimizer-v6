from smart_warehouse.allocator import ExplainableAllocator
from smart_warehouse.entities import Robot, Task
from smart_warehouse.warehouse import WarehouseMap


def test_allocator_selects_closer_robot() -> None:
    warehouse = WarehouseMap.create_default()
    task = Task("T-1", 0, "A", (2, 3))
    robots = [Robot("R-1", (2, 2)), Robot("R-2", (18, 18))]
    decision = ExplainableAllocator().allocate(task, robots, warehouse, 0)
    assert decision is not None
    assert decision.robot_id == "R-1"
