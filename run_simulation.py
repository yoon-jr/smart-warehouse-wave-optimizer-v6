from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pygame

from smart_warehouse.config import DATA_DIR, SimulationConfig
from smart_warehouse.entities import RobotStatus, TaskStatus
from smart_warehouse.footwear_replay import FootwearTaskReplay
from smart_warehouse.simulation import SimulationEngine

BACKGROUND = (245, 247, 250)
GRID = (210, 215, 222)
SHELF = (70, 76, 85)
TEXT = (25, 30, 38)
MUTED = (94, 101, 112)
WHITE = (255, 255, 255)
ROBOT_COLORS = {
    RobotStatus.IDLE: (40, 120, 220),
    RobotStatus.REPOSITIONING: (143, 93, 183),
    RobotStatus.MOVING_TO_TASK: (25, 150, 100),
    RobotStatus.WORKING: (245, 150, 35),
}
TASK_COLORS = {
    TaskStatus.WAITING: (220, 50, 60),
    TaskStatus.ASSIGNED: (230, 110, 45),
    TaskStatus.IN_PROGRESS: (245, 190, 50),
}
ZONE_TINTS = {
    "A": (232, 242, 255),
    "B": (238, 250, 239),
    "C": (252, 242, 253),
    "D": (255, 248, 229),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-policy Footwear warehouse replay")
    parser.add_argument("--mode", choices=["reactive", "predictive", "hybrid"], default="hybrid")
    parser.add_argument("--tasks", type=Path, default=DATA_DIR / "footwear_replay_tasks.csv")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "footwear_manifest.json")
    parser.add_argument("--max-ticks", type=int, default=0)
    return parser.parse_args()


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, x: int, y: int, color=TEXT) -> None:
    surface.blit(font.render(str(text), True, color), (x, y))


def draw_simulation(screen: pygame.Surface, engine: SimulationEngine, config: SimulationConfig) -> None:
    screen.fill(BACKGROUND)
    grid_width_px = config.grid_width * config.cell_size

    for y in range(config.grid_height):
        for x in range(config.grid_width):
            zone_id = engine.warehouse.zone_for((x, y))
            rect = pygame.Rect(x * config.cell_size, y * config.cell_size, config.cell_size, config.cell_size)
            pygame.draw.rect(screen, ZONE_TINTS[zone_id], rect)
            pygame.draw.rect(screen, GRID, rect, 1)
            if (x, y) in engine.warehouse.obstacles:
                pygame.draw.rect(screen, SHELF, rect.inflate(-4, -4), border_radius=3)

    for robot in engine.robots:
        if robot.path:
            points = [robot.position, *robot.path]
            pixel_points = [
                (x * config.cell_size + config.cell_size // 2, y * config.cell_size + config.cell_size // 2)
                for x, y in points
            ]
            if len(pixel_points) > 1:
                pygame.draw.lines(screen, ROBOT_COLORS[robot.status], False, pixel_points, 2)

    for task in engine.tasks.values():
        if task.status == TaskStatus.COMPLETED:
            continue
        x, y = task.position
        center = (x * config.cell_size + config.cell_size // 2, y * config.cell_size + config.cell_size // 2)
        pygame.draw.rect(
            screen,
            TASK_COLORS[task.status],
            pygame.Rect(center[0] - 7, center[1] - 7, 14, 14),
            border_radius=2,
        )

    small_font = pygame.font.SysFont("arial", 13)
    for robot in engine.robots:
        x, y = robot.position
        center = (x * config.cell_size + config.cell_size // 2, y * config.cell_size + config.cell_size // 2)
        pygame.draw.circle(screen, ROBOT_COLORS[robot.status], center, 11)
        label = small_font.render(robot.robot_id.split("-")[-1], True, WHITE)
        screen.blit(label, label.get_rect(center=center))

    panel_x = grid_width_px + 20
    font = pygame.font.SysFont("arial", 18)
    title_font = pygame.font.SysFont("arial", 24, bold=True)
    draw_text(screen, title_font, "Footwear Warehouse Replay", panel_x, 20)
    draw_text(screen, font, f"Controller: {engine.mode.upper()}", panel_x, 58)
    if engine.mode == "hybrid":
        draw_text(screen, font, f"Active policy: {engine.active_policy}", panel_x, 86)
    draw_text(screen, font, f"Model: {engine.predictor.model_name}", panel_x, 114)

    metrics = engine.metrics_summary()
    y = 155
    for label, value in [
        ("Completed", metrics["completed_tasks"]),
        ("Waiting", metrics["waiting_tasks"]),
        ("Average wait", metrics["average_wait_time"]),
        ("Total distance", metrics["total_distance"]),
        ("Reposition distance", metrics["reposition_distance"]),
    ]:
        draw_text(screen, font, f"{label}: {value}", panel_x, y)
        y += 29

    y += 8
    draw_text(screen, font, "How to read the map", panel_x, y)
    y += 29
    legend = [
        ("Red square", "waiting task"),
        ("Blue circle", "idle robot"),
        ("Green", "robot/route to released task"),
        ("Purple", "predictive repositioning"),
        ("Dark block", "shelf/obstacle"),
    ]
    for label, meaning in legend:
        draw_text(screen, small_font, f"{label} = {meaning}", panel_x, y, MUTED)
        y += 21

    if engine.mode == "hybrid" and engine.policy_decision:
        y += 8
        draw_text(screen, font, "Why this mode?", panel_x, y)
        y += 27
        reason = engine.policy_decision.reason
        words = reason.split()
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if small_font.size(candidate)[0] < config.panel_width - 35:
                line = candidate
            else:
                draw_text(screen, small_font, line, panel_x, y, MUTED)
                y += 20
                line = word
        if line:
            draw_text(screen, small_font, line, panel_x, y, MUTED)

    draw_text(
        screen,
        small_font,
        "Use run_comparison.py for the full Reactive vs Adaptive Hybrid demo.",
        panel_x,
        config.grid_height * config.cell_size - 52,
        MUTED,
    )
    draw_text(screen, small_font, "SPACE pause | R reset | ESC quit", panel_x, config.grid_height * config.cell_size - 28)


def main() -> int:
    args = parse_args()
    config = SimulationConfig()
    replay = FootwearTaskReplay(args.tasks, manifest_path=args.manifest, random_seed=config.random_seed)
    if replay.simulation_minutes_per_tick is not None:
        config = replace(config, simulation_minutes_per_tick=replay.simulation_minutes_per_tick)
    engine = SimulationEngine(config, args.mode, initial_minute=replay.start_minute)

    pygame.init()
    screen = pygame.display.set_mode(
        (config.grid_width * config.cell_size + config.panel_width, config.grid_height * config.cell_size)
    )
    pygame.display.set_caption("Footwear Smart Warehouse")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    engine.paused = not engine.paused
                elif event.key == pygame.K_r:
                    engine = SimulationEngine(config, args.mode, initial_minute=replay.start_minute)

        next_tick = engine.tick_count + 1
        engine.step(external_tasks=replay.tasks_for_tick(next_tick))
        draw_simulation(screen, engine, config)
        pygame.display.flip()
        clock.tick(config.fps)

        if args.max_ticks > 0 and engine.tick_count >= args.max_ticks:
            running = False

    engine.store.write_state(engine.snapshot())
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
