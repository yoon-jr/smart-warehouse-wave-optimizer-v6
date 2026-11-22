from smart_warehouse.entities import Robot
from smart_warehouse.reservations import (
    ReservationTable,
    detect_conflicts,
    space_time_astar,
)
from smart_warehouse.warehouse import WarehouseMap


def test_space_time_astar_avoids_reserved_vertex() -> None:
    warehouse = WarehouseMap(width=3, height=2, obstacles=set(), zones={})
    table = ReservationTable(horizon=8)
    table.reserve_path((1, 0), [], hold_steps=8)
    path = space_time_astar(
        warehouse,
        (0, 0),
        (2, 0),
        table,
        max_time=8,
        goal_hold_steps=1,
    )
    assert path
    assert path[-1] == (2, 0)
    assert (1, 0) not in path


def test_detect_conflicts_finds_vertex_and_edge_swap() -> None:
    current = {"R-1": (0, 0), "R-2": (1, 0), "R-3": (2, 0)}
    proposed = {"R-1": (1, 0), "R-2": (0, 0), "R-3": (1, 0)}
    vertex, edge = detect_conflicts(current, proposed)
    assert vertex
    assert edge


def test_default_fleet_starts_on_unique_vertices() -> None:
    from smart_warehouse.config import SimulationConfig
    from smart_warehouse.simulation import SimulationEngine

    engine = SimulationEngine(SimulationConfig(robot_count=8), persist=False)
    positions = [robot.position for robot in engine.robots]
    assert len(positions) == len(set(positions))


def test_existing_route_terminal_is_reserved_through_horizon() -> None:
    from smart_warehouse.reservations import build_reservation_table

    robot = Robot(robot_id="R-2", position=(0, 0), path=[(1, 0), (2, 0)])
    table = build_reservation_table(
        [robot],
        exclude_robot_id=None,
        horizon=8,
        hold_steps=1,
    )
    assert table.vertex_reserved((2, 0), 2)
    assert table.vertex_reserved((2, 0), 8)
