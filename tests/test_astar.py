from smart_warehouse.astar import astar
from smart_warehouse.warehouse import WarehouseMap


def test_astar_reaches_goal_without_crossing_obstacles() -> None:
    warehouse = WarehouseMap.create_default()
    start = warehouse.zones["A"].waiting_position
    goal = warehouse.zones["D"].waiting_position
    path = astar(warehouse, start, goal)
    assert path
    assert path[-1] == goal
    assert all(warehouse.walkable(point) for point in path)
