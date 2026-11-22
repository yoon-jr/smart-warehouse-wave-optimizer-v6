from __future__ import annotations

"""Side-by-side Lifelong MAPFUA demo.

The left engine keeps unassigned robots stationary. The right engine runs one
of the v5 idle-robot policies (heuristic or trained VDN). Both engines receive
the exact same WMS-derived task rows at the exact same ticks.
"""

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

# pygame is loaded lazily in main so --help and static checks work even when
# the optional visual dependency is not installed.
pygame = None

from smart_warehouse.config import DATA_DIR, MODEL_DIR, SimulationConfig
from smart_warehouse.entities import RobotStatus, TaskStatus
from smart_warehouse.footwear_replay import FootwearTaskReplay
from smart_warehouse.idle_policy import ExternalJointPolicy, IdleStayPolicy, policy_from_name
from smart_warehouse.mapfua_env import build_agent_observations, temporal_splits
from smart_warehouse.marl_vdn import VDNLearner
from smart_warehouse.simulation import SimulationEngine

BACKGROUND = (242, 245, 249)
PANEL = (255, 255, 255)
GRID = (207, 214, 224)
SHELF = (64, 70, 80)
TEXT = (28, 33, 41)
MUTED = (91, 99, 111)
WHITE = (255, 255, 255)
BORDER = (216, 222, 231)
POSITIVE = (24, 142, 91)
NEGATIVE = (215, 65, 72)
PARKING = (120, 126, 138)
ZONE_TINTS = {
    "A": (229, 240, 255),
    "B": (231, 248, 235),
    "C": (248, 234, 250),
    "D": (255, 246, 223),
}
ROBOT_COLORS = {
    RobotStatus.IDLE: (42, 121, 219),
    RobotStatus.REPOSITIONING: (138, 85, 181),
    RobotStatus.MOVING_TO_TASK: (24, 151, 103),
    RobotStatus.WORKING: (242, 150, 38),
}
TASK_COLORS = {
    TaskStatus.WAITING: (220, 54, 64),
    TaskStatus.ASSIGNED: (229, 111, 47),
    TaskStatus.IN_PROGRESS: (244, 189, 51),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visual comparison of Idle-Stay and a Lifelong MAPFUA idle-robot "
            "policy on the same chronological WMS task window."
        )
    )
    parser.add_argument("--tasks", type=Path, default=DATA_DIR / "footwear_replay_tasks.csv")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "footwear_manifest.json")
    parser.add_argument(
        "--policy",
        choices=["nearest_parking", "density_clear", "forecast_rule", "vdn_marl"],
        default="density_clear",
    )
    parser.add_argument("--vdn-model", type=Path, default=MODEL_DIR / "mapfua_vdn.pt")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--start-tick", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robot-count", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def initial_minute(replay: FootwearTaskReplay, start_tick: int, tick_minutes: float) -> float:
    future = replay.frame[replay.frame["release_tick"].astype(int) >= start_tick]
    if future.empty:
        return replay.start_minute
    return float(future.iloc[0]["created_minute"]) - tick_minutes


def choose_start_tick(
    replay: FootwearTaskReplay,
    split_name: str,
    max_ticks: int,
    seed: int,
    fixed: int | None,
) -> int:
    if fixed is not None:
        if not replay.start_tick <= fixed <= replay.max_tick:
            raise ValueError(
                f"--start-tick={fixed} is outside replay range "
                f"[{replay.start_tick}, {replay.max_tick}]"
            )
        return int(fixed)
    split = temporal_splits(replay.frame)[split_name]
    latest = max(split.start_tick, split.end_tick - max_ticks + 1)
    candidates = sorted(
        {
            int(value)
            for value in replay.frame["release_tick"].astype(int)
            if split.start_tick <= int(value) <= latest
        }
    )
    if not candidates:
        return int(split.start_tick)
    import random

    return int(random.Random(seed).choice(candidates))


def make_engines(
    config: SimulationConfig,
    replay: FootwearTaskReplay,
    start_tick: int,
    candidate_policy: str,
) -> tuple[SimulationEngine, SimulationEngine]:
    start_minute = initial_minute(replay, start_tick, config.simulation_minutes_per_tick)
    baseline = SimulationEngine(
        config,
        mode="reactive",
        persist=False,
        initial_minute=start_minute,
        idle_policy=IdleStayPolicy(),
    )
    candidate_idle_policy = (
        ExternalJointPolicy() if candidate_policy == "vdn_marl" else policy_from_name(candidate_policy)
    )
    candidate = SimulationEngine(
        config,
        mode="reactive",
        persist=False,
        initial_minute=start_minute,
        idle_policy=candidate_idle_policy,
    )
    return baseline, candidate


def draw_text(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: object,
    x: int,
    y: int,
    color: tuple[int, int, int] = TEXT,
) -> None:
    screen.blit(font.render(str(text), True, color), (x, y))


def draw_wrapped(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    rect: pygame.Rect,
    color: tuple[int, int, int] = MUTED,
    max_lines: int = 3,
) -> int:
    words = str(text).split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if font.size(candidate)[0] <= rect.width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
            if len(lines) >= max_lines:
                break
    if line and len(lines) < max_lines:
        lines.append(line)
    y = rect.y
    for item in lines[:max_lines]:
        draw_text(screen, font, item, rect.x, y, color)
        y += font.get_linesize()
    return y


def draw_engine(
    screen: pygame.Surface,
    engine: SimulationEngine,
    origin: tuple[int, int],
    cell: int,
    title: str,
) -> None:
    ox, oy = origin
    title_font = pygame.font.SysFont("arial", 18, bold=True)
    small = pygame.font.SysFont("arial", 11)
    metric = pygame.font.SysFont("arial", 12)
    draw_text(screen, title_font, title, ox, oy - 28)

    parking = set(engine.warehouse.parking_nodes())
    for y in range(engine.warehouse.height):
        for x in range(engine.warehouse.width):
            rect = pygame.Rect(ox + x * cell, oy + y * cell, cell, cell)
            zone = engine.warehouse.zone_for((x, y))
            pygame.draw.rect(screen, ZONE_TINTS[zone], rect)
            pygame.draw.rect(screen, GRID, rect, 1)
            if (x, y) in engine.warehouse.obstacles:
                pygame.draw.rect(screen, SHELF, rect.inflate(-3, -3), border_radius=2)
            elif (x, y) in parking:
                pygame.draw.circle(screen, PARKING, rect.center, max(1, cell // 8), 1)

    for robot in engine.robots:
        if robot.path:
            points = [robot.position, *robot.path[:20]]
            pixels = [(ox + x * cell + cell // 2, oy + y * cell + cell // 2) for x, y in points]
            if len(pixels) > 1:
                pygame.draw.lines(screen, ROBOT_COLORS[robot.status], False, pixels, 2)

    grouped: dict[tuple[tuple[int, int], TaskStatus], int] = defaultdict(int)
    for task in engine.tasks.values():
        if task.status != TaskStatus.COMPLETED:
            grouped[(task.position, task.status)] += 1
    priority = {TaskStatus.WAITING: 3, TaskStatus.ASSIGNED: 2, TaskStatus.IN_PROGRESS: 1}
    by_position: dict[tuple[int, int], list[tuple[TaskStatus, int]]] = defaultdict(list)
    for (position, status), count in grouped.items():
        by_position[position].append((status, count))
    for (x, y), groups in by_position.items():
        status, count = max(groups, key=lambda item: priority[item[0]])
        center = (ox + x * cell + cell // 2, oy + y * cell + cell // 2)
        size = 7 if count == 1 else 10
        pygame.draw.rect(
            screen,
            TASK_COLORS[status],
            pygame.Rect(center[0] - size // 2, center[1] - size // 2, size, size),
            border_radius=2,
        )

    for robot in engine.robots:
        x, y = robot.position
        center = (ox + x * cell + cell // 2, oy + y * cell + cell // 2)
        pygame.draw.circle(screen, ROBOT_COLORS[robot.status], center, max(5, cell // 3))
        label = small.render(robot.robot_id[-2:], True, WHITE)
        screen.blit(label, label.get_rect(center=center))

    metrics = engine.metrics_summary()
    base_y = oy + engine.warehouse.height * cell + 8
    draw_text(
        screen,
        metric,
        (
            f"Done {metrics['completed_tasks']} | Waiting {metrics['waiting_tasks']} | "
            f"Avg/P95 wait {metrics['average_wait_time']:.1f}/{metrics['p95_wait_time']:.1f}"
        ),
        ox,
        base_y,
    )
    draw_text(
        screen,
        metric,
        (
            f"Distance {metrics['total_distance']} | Idle {metrics['idle_reposition_distance']} | "
            f"Blocking proxy {metrics['blocking_pressure_ticks']:.1f}"
        ),
        ox,
        base_y + 18,
    )
    draw_text(
        screen,
        metric,
        (
            f"Idle moves {metrics['idle_move_events']} | Useful est. "
            f"{metrics['estimated_useful_idle_move_rate_pct']:.1f}% | "
            f"Shield waits {metrics['collision_shield_waits']}"
        ),
        ox,
        base_y + 36,
    )


def reduction(baseline: float, contender: float) -> float:
    if abs(baseline) < 1e-9:
        return 0.0
    return 100.0 * (baseline - contender) / baseline


def draw_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    baseline: SimulationEngine,
    candidate: SimulationEngine,
    policy_name: str,
    start_tick: int,
    local_tick: int,
) -> None:
    pygame.draw.rect(screen, PANEL, rect, border_radius=10)
    pygame.draw.rect(screen, BORDER, rect, 1, border_radius=10)
    title = pygame.font.SysFont("arial", 20, bold=True)
    heading = pygame.font.SysFont("arial", 13, bold=True)
    font = pygame.font.SysFont("arial", 12)
    small = pygame.font.SysFont("arial", 11)
    x = rect.x + 16
    y = rect.y + 15
    draw_text(screen, title, "LIFELONG MAPFUA", x, y)
    y += 32
    draw_text(screen, font, f"Global tick {start_tick + local_tick - 1} | local {local_tick}", x, y, MUTED)
    y += 23
    draw_text(screen, heading, "RESEARCH QUESTION", x, y)
    y = draw_wrapped(
        screen,
        font,
        "When should unassigned robots stay, clear a bottleneck, or reposition?",
        pygame.Rect(x, y + 20, rect.width - 32, 54),
        TEXT,
        3,
    ) + 10

    bm = baseline.metrics_summary()
    cm = candidate.metrics_summary()
    draw_text(screen, heading, f"{policy_name.upper()} VS IDLE-STAY", x, y)
    y += 24
    comparisons = [
        ("Average wait reduction", reduction(float(bm["average_wait_time"]), float(cm["average_wait_time"]))),
        ("P95 wait reduction", reduction(float(bm["p95_wait_time"]), float(cm["p95_wait_time"]))),
        ("Distance reduction", reduction(float(bm["total_distance"]), float(cm["total_distance"]))),
        (
            "Blocking-proxy reduction",
            reduction(float(bm["blocking_pressure_ticks"]), float(cm["blocking_pressure_ticks"])),
        ),
    ]
    for label, value in comparisons:
        draw_text(screen, font, label, x, y, MUTED)
        color = POSITIVE if value >= 0 else NEGATIVE
        draw_text(screen, heading, f"{value:+.1f}%", rect.right - 74, y - 1, color)
        y += 23

    y += 5
    draw_text(screen, heading, "LATEST IDLE-ROBOT DECISIONS", x, y)
    y += 22
    decisions = candidate.last_idle_decisions[-4:]
    if not decisions:
        draw_text(screen, small, "No high-level decision recorded yet.", x, y, MUTED)
        y += 22
    for decision in decisions:
        draw_text(
            screen,
            small,
            f"{decision.robot_id}: {decision.action.name} ({decision.purpose})",
            x,
            y,
            TEXT,
        )
        y = draw_wrapped(
            screen,
            small,
            decision.reason,
            pygame.Rect(x + 8, y + 16, rect.width - 48, 40),
            MUTED,
            2,
        ) + 5

    y = max(y, rect.bottom - 116)
    draw_text(screen, heading, "LEGEND", x, y)
    y += 20
    draw_text(screen, small, "Blue idle | Green task route | Purple idle move", x, y, MUTED)
    y += 18
    draw_text(screen, small, "Red task | Dark shelf | Grey dot parking node", x, y, MUTED)
    y += 24
    draw_text(screen, small, "SPACE pause | R reset | ESC quit", x, y, TEXT)
    draw_text(screen, small, "Metrics marked 'proxy/estimated' are diagnostic, not ground truth.", x, y + 18, MUTED)


def main() -> int:
    args = parse_args()
    global pygame
    try:
        import pygame as pygame_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pygame is required for run_mapfua_demo.py. "
            "Install requirements-marl.txt first."
        ) from exc
    pygame = pygame_module

    if not args.tasks.exists():
        raise FileNotFoundError(
            f"Prepared task replay not found: {args.tasks}. Run prepare_footwear_data first."
        )
    replay = FootwearTaskReplay(args.tasks, manifest_path=args.manifest, random_seed=args.seed)
    config = replace(
        SimulationConfig(),
        robot_count=args.robot_count,
        random_seed=args.seed,
        fps=args.fps,
        snapshot_interval_ticks=max(1, args.max_ticks + 1),
        simulation_minutes_per_tick=(
            replay.simulation_minutes_per_tick
            if replay.simulation_minutes_per_tick is not None
            else SimulationConfig().simulation_minutes_per_tick
        ),
    )
    start_tick = choose_start_tick(
        replay,
        args.split,
        args.max_ticks,
        args.seed,
        args.start_tick,
    )

    learner: VDNLearner | None = None
    learner_include_forecast = False
    if args.policy == "vdn_marl":
        if not args.vdn_model.exists():
            raise FileNotFoundError(f"VDN model not found: {args.vdn_model}")
        learner = VDNLearner.load(args.vdn_model)
        if learner.robot_count != args.robot_count:
            raise ValueError(
                f"VDN model expects {learner.robot_count} robots, "
                f"but --robot-count={args.robot_count}"
            )
        learner_include_forecast = bool(learner.metadata.get("include_forecast", False))

    baseline, candidate = make_engines(config, replay, start_tick, args.policy)

    pygame.init()
    cell = 18
    map_width = config.grid_width * cell
    map_height = config.grid_height * cell
    margin = 24
    gap = 24
    panel_width = 340
    width = margin + map_width + gap + map_width + gap + panel_width + margin
    height = max(585, margin + 28 + map_height + 62 + margin)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Lifelong MAPFUA: Idle-Stay vs Idle-Robot Policy")
    clock = pygame.time.Clock()
    panel_rect = pygame.Rect(width - panel_width - margin, margin, panel_width, height - 2 * margin)

    paused = False
    running = True
    local_tick = 0
    last_wall = time.perf_counter()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    baseline, candidate = make_engines(config, replay, start_tick, args.policy)
                    local_tick = 0

        if not paused and local_tick < args.max_ticks:
            global_tick = start_tick + local_tick
            events = replay.tasks_for_tick(global_tick)
            if args.policy == "vdn_marl" and (
                local_tick == 0
                or local_tick % config.idle_decision_interval_ticks == 0
            ):
                assert learner is not None
                decision_started = time.perf_counter()
                if learner_include_forecast:
                    candidate.refresh_predictions()
                observations, masks = build_agent_observations(
                    candidate,
                    include_forecast=learner_include_forecast,
                )
                actions = learner.select_actions(observations, masks, explore=False)
                candidate.apply_idle_actions(
                    dict(zip(sorted(robot.robot_id for robot in candidate.robots), actions.tolist()))
                )
                candidate.record_idle_decision_latency(
                    (time.perf_counter() - decision_started) * 1000.0
                )
            baseline.step(external_tasks=events)
            candidate.step(external_tasks=events)
            local_tick += 1

        screen.fill(BACKGROUND)
        first_origin = (margin, margin + 30)
        second_origin = (margin + map_width + gap, margin + 30)
        draw_engine(screen, baseline, first_origin, cell, "BASELINE: IDLE-STAY")
        draw_engine(screen, candidate, second_origin, cell, f"CANDIDATE: {args.policy.upper()}")
        draw_panel(
            screen,
            panel_rect,
            baseline,
            candidate,
            args.policy,
            start_tick,
            max(1, local_tick),
        )
        status_font = pygame.font.SysFont("arial", 11)
        elapsed = max(1e-9, time.perf_counter() - last_wall)
        draw_text(
            screen,
            status_font,
            f"Window start {start_tick} | {args.split} split | {args.robot_count} robots | UI FPS {clock.get_fps():.1f}",
            margin,
            height - 18,
            MUTED,
        )
        last_wall += elapsed
        pygame.display.flip()
        clock.tick(config.fps)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
