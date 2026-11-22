from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pygame

from smart_warehouse.comparison import ComparisonRunner
from smart_warehouse.config import DATA_DIR, SimulationConfig
from smart_warehouse.entities import RobotStatus, TaskStatus

BACKGROUND = (244, 246, 250)
GRID = (205, 211, 220)
SHELF = (64, 70, 80)
TEXT = (26, 31, 39)
MUTED = (92, 100, 112)
WHITE = (255, 255, 255)
PANEL = (255, 255, 255)
BORDER = (217, 222, 230)
ACCENT = (20, 110, 230)
SURGE = (220, 55, 65)
POSITIVE = (20, 145, 90)
WARNING = (230, 145, 20)
PURPLE = (133, 82, 180)
ROBOT_COLORS = {
    RobotStatus.IDLE: (40, 120, 220),
    RobotStatus.REPOSITIONING: PURPLE,
    RobotStatus.MOVING_TO_TASK: (25, 150, 100),
    RobotStatus.WORKING: (245, 150, 35),
}
TASK_COLORS = {
    TaskStatus.WAITING: (220, 50, 60),
    TaskStatus.ASSIGNED: (230, 110, 45),
    TaskStatus.IN_PROGRESS: (245, 190, 50),
}
ZONE_BASE = {
    "A": (229, 240, 255),
    "B": (231, 248, 235),
    "C": (248, 234, 250),
    "D": (255, 246, 223),
}
LEVEL_COLORS = {
    "NORMAL": POSITIVE,
    "BUSY": WARNING,
    "OVERLOADED": SURGE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reactive baseline vs threshold-based Adaptive Hybrid warehouse demo."
    )
    parser.add_argument("--tasks", type=Path, default=DATA_DIR / "footwear_replay_tasks.csv")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "footwear_manifest.json")
    parser.add_argument("--max-ticks", type=int, default=0)
    parser.add_argument("--surge-tick", type=int, default=0)
    parser.add_argument("--surge-zone", choices=["A", "B", "C", "D"], default="C")
    parser.add_argument("--surge-count", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def blend(first: tuple[int, int, int], second: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = min(1.0, max(0.0, amount))
    return tuple(round((1 - amount) * a + amount * b) for a, b in zip(first, second))


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = TEXT,
) -> None:
    surface.blit(font.render(str(text), True, color), (x, y))


def draw_wrapped_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    rect: pygame.Rect,
    color: tuple[int, int, int] = TEXT,
    *,
    line_spacing: int = 3,
    max_lines: int | None = None,
) -> int:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if font.size(candidate)[0] <= rect.width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if max_lines is not None:
        lines = lines[:max_lines]
    y = rect.y
    for line in lines:
        surface.blit(font.render(line, True, color), (rect.x, y))
        y += font.get_linesize() + line_spacing
    return y


def draw_badge(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    rect: pygame.Rect,
    color: tuple[int, int, int],
) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=8)
    label = font.render(text, True, WHITE)
    surface.blit(label, label.get_rect(center=rect.center))


def draw_engine(
    screen: pygame.Surface,
    engine,
    *,
    origin_x: int,
    origin_y: int,
    cell_size: int,
    title: str,
    subtitle: str,
    show_policy: bool = False,
) -> None:
    title_font = pygame.font.SysFont("arial", 23, bold=True)
    subtitle_font = pygame.font.SysFont("arial", 12)
    metric_font = pygame.font.SysFont("arial", 15)
    small_font = pygame.font.SysFont("arial", 11, bold=True)
    zone_font = pygame.font.SysFont("arial", 10, bold=True)
    draw_text(screen, title_font, title, origin_x, 14)
    draw_text(screen, subtitle_font, subtitle, origin_x, 42, MUTED)

    if show_policy:
        policy = engine.active_policy
        color = PURPLE if policy == "PREDICTIVE" else ACCENT
        draw_badge(
            screen,
            small_font,
            f"ACTIVE: {policy}",
            pygame.Rect(origin_x + engine.warehouse.width * cell_size - 126, 17, 124, 24),
            color,
        )

    heat = engine.demand_heatmap_values()
    max_heat = max(max(heat.values(), default=0.0), 1.0)
    for y in range(engine.warehouse.height):
        for x in range(engine.warehouse.width):
            zone_id = engine.warehouse.zone_for((x, y))
            intensity = min(0.62, 0.62 * float(heat.get(zone_id, 0.0)) / max_heat)
            color = blend(ZONE_BASE[zone_id], (255, 77, 70), intensity)
            rect = pygame.Rect(
                origin_x + x * cell_size,
                origin_y + y * cell_size,
                cell_size,
                cell_size,
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID, rect, 1)
            if (x, y) in engine.warehouse.obstacles:
                pygame.draw.rect(screen, SHELF, rect.inflate(-4, -4), border_radius=2)

    queue = engine.queue_counts()
    for zone_id, zone in engine.warehouse.zones.items():
        label_x = origin_x + zone.x_min * cell_size + 4
        label_y = origin_y + zone.y_min * cell_size + 4
        forecast = float(engine.predictions.get(zone_id, 0.0))
        label = f"ZONE {zone_id}  waiting {queue[zone_id]}  forecast {forecast:.1f}"
        label_surface = zone_font.render(label, True, TEXT)
        bg_rect = label_surface.get_rect(topleft=(label_x, label_y)).inflate(8, 5)
        overlay = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 205))
        screen.blit(overlay, bg_rect.topleft)
        screen.blit(label_surface, (label_x, label_y))

    for robot in engine.robots:
        if not robot.path:
            continue
        points = [robot.position, *robot.path]
        pixel_points = [
            (
                origin_x + x * cell_size + cell_size // 2,
                origin_y + y * cell_size + cell_size // 2,
            )
            for x, y in points
        ]
        if len(pixel_points) > 1:
            pygame.draw.lines(screen, ROBOT_COLORS[robot.status], False, pixel_points, 2)

    # Aggregate overlapping tasks. This avoids hundreds of identical red marks
    # hiding the underlying map during a stress test.
    task_groups: dict[tuple[tuple[int, int], TaskStatus], int] = defaultdict(int)
    for task in engine.tasks.values():
        if task.status != TaskStatus.COMPLETED:
            task_groups[(task.position, task.status)] += 1
    status_priority = {TaskStatus.WAITING: 3, TaskStatus.ASSIGNED: 2, TaskStatus.IN_PROGRESS: 1}
    location_groups: dict[tuple[int, int], list[tuple[TaskStatus, int]]] = defaultdict(list)
    for (position, status), count in task_groups.items():
        location_groups[position].append((status, count))
    for (x, y), groups in location_groups.items():
        status, count = max(groups, key=lambda item: status_priority[item[0]])
        center = (
            origin_x + x * cell_size + cell_size // 2,
            origin_y + y * cell_size + cell_size // 2,
        )
        size = 10 if count == 1 else 14
        pygame.draw.rect(
            screen,
            TASK_COLORS[status],
            pygame.Rect(center[0] - size // 2, center[1] - size // 2, size, size),
            border_radius=2,
        )
        if count > 1:
            count_text = small_font.render(str(min(count, 99)), True, WHITE)
            screen.blit(count_text, count_text.get_rect(center=center))

    for robot in engine.robots:
        x, y = robot.position
        center = (
            origin_x + x * cell_size + cell_size // 2,
            origin_y + y * cell_size + cell_size // 2,
        )
        pygame.draw.circle(screen, ROBOT_COLORS[robot.status], center, max(7, cell_size // 3))
        label = small_font.render(robot.robot_id.split("-")[-1], True, WHITE)
        screen.blit(label, label.get_rect(center=center))

    metrics = engine.metrics_summary()
    baseline_y = origin_y + engine.warehouse.height * cell_size + 10
    draw_text(
        screen,
        metric_font,
        f"Done {metrics['completed_tasks']}   Waiting {metrics['waiting_tasks']}   Avg wait {metrics['average_wait_time']:.1f}",
        origin_x,
        baseline_y,
    )
    draw_text(
        screen,
        metric_font,
        f"Throughput {metrics['throughput_per_hour']:.1f}/h   Total distance {metrics['total_distance']}",
        origin_x,
        baseline_y + 21,
    )
    if show_policy:
        draw_text(
            screen,
            subtitle_font,
            f"Policy use: predictive {metrics['predictive_policy_pct']:.0f}% · reactive {metrics['reactive_policy_pct']:.0f}%",
            origin_x,
            baseline_y + 44,
            MUTED,
        )


def draw_impact_row(
    screen: pygame.Surface,
    label_font: pygame.font.Font,
    value_font: pygame.font.Font,
    x: int,
    y: int,
    label: str,
    value: float,
) -> None:
    draw_text(screen, label_font, label, x, y, MUTED)
    color = POSITIVE if value >= 0 else SURGE
    draw_text(screen, value_font, f"{value:+.1f}%", x + 225, y - 4, color)


def draw_panel(
    screen: pygame.Surface,
    runner: ComparisonRunner,
    panel_rect: pygame.Rect,
    button_rect: pygame.Rect,
) -> None:
    title_font = pygame.font.SysFont("arial", 22, bold=True)
    heading_font = pygame.font.SysFont("arial", 14, bold=True)
    font = pygame.font.SysFont("arial", 13)
    small_font = pygame.font.SysFont("arial", 11)
    value_font = pygame.font.SysFont("arial", 16, bold=True)

    pygame.draw.rect(screen, PANEL, panel_rect, border_radius=10)
    pygame.draw.rect(screen, BORDER, panel_rect, 1, border_radius=10)
    x = panel_rect.x + 18
    width = panel_rect.width - 36
    draw_text(screen, title_font, "LIVE EXPLAINER", x, panel_rect.y + 14)
    draw_text(screen, small_font, f"Tick {runner.tick_count} · {runner.data_source_label}", x, panel_rect.y + 44, MUTED)

    situation = runner.situation_summary()
    level_color = LEVEL_COLORS[situation["level"]]
    situation_rect = pygame.Rect(x, panel_rect.y + 68, width, 112)
    pygame.draw.rect(screen, blend(WHITE, level_color, 0.09), situation_rect, border_radius=8)
    pygame.draw.rect(screen, level_color, situation_rect, 2, border_radius=8)
    draw_badge(
        screen,
        small_font,
        situation["level"],
        pygame.Rect(situation_rect.x + 10, situation_rect.y + 9, 92, 22),
        level_color,
    )
    draw_text(screen, heading_font, situation["headline"], situation_rect.x + 112, situation_rect.y + 10)
    draw_wrapped_text(
        screen,
        font,
        situation["summary"],
        pygame.Rect(situation_rect.x + 10, situation_rect.y + 39, situation_rect.width - 20, 66),
        MUTED,
        max_lines=3,
    )

    policy_y = situation_rect.bottom + 12
    draw_text(screen, heading_font, "WHY THE HYBRID CHOSE THIS MODE", x, policy_y)
    policy_color = PURPLE if situation["active_policy"] == "PREDICTIVE" else ACCENT
    draw_badge(
        screen,
        heading_font,
        situation["active_policy"],
        pygame.Rect(x, policy_y + 22, 118, 30),
        policy_color,
    )
    reason_y = draw_wrapped_text(
        screen,
        font,
        situation["policy_reason"],
        pygame.Rect(x + 130, policy_y + 20, width - 130, 55),
        TEXT,
        max_lines=3,
    )
    action_y = max(policy_y + 62, reason_y + 4)
    draw_text(screen, heading_font, "CURRENT ACTION", x, action_y)
    draw_wrapped_text(
        screen,
        font,
        situation["current_action"],
        pygame.Rect(x + 112, action_y - 1, width - 112, 42),
        MUTED,
        max_lines=2,
    )

    impact_y = action_y + 50
    draw_text(screen, heading_font, "HYBRID IMPACT VS REACTIVE BASELINE", x, impact_y)
    improvements = runner.improvements()
    draw_impact_row(screen, font, value_font, x, impact_y + 27, "Average wait reduction", improvements["wait_time_reduction_pct"])
    draw_impact_row(screen, font, value_font, x, impact_y + 51, "P95 wait reduction", improvements["p95_wait_reduction_pct"])
    draw_impact_row(screen, font, value_font, x, impact_y + 75, "Throughput increase", improvements["throughput_increase_pct"])
    draw_impact_row(screen, font, value_font, x, impact_y + 99, "Task travel reduction", improvements["task_distance_reduction_pct"])

    pygame.draw.rect(screen, SURGE, button_rect, border_radius=8)
    label = title_font.render("ORDER SURGE", True, WHITE)
    screen.blit(label, label.get_rect(center=button_rect.center))
    draw_text(screen, small_font, "40-tick advance notice; identical tasks then go to both systems", x, button_rect.bottom + 7, MUTED)

    legend_y = button_rect.bottom + 34
    draw_text(screen, heading_font, "HOW TO READ THE MAP", x, legend_y)
    legend_items = [
        (TASK_COLORS[TaskStatus.WAITING], "Red square = released task waiting for a robot"),
        (TASK_COLORS[TaskStatus.ASSIGNED], "Orange/yellow square = task assigned or being processed"),
        (ROBOT_COLORS[RobotStatus.IDLE], "Blue circle = idle robot"),
        (ROBOT_COLORS[RobotStatus.MOVING_TO_TASK], "Green circle/line = robot and A* route to task"),
        (ROBOT_COLORS[RobotStatus.REPOSITIONING], "Purple circle/line = predictive repositioning"),
        (ROBOT_COLORS[RobotStatus.WORKING], "Orange circle = robot processing a task"),
        (SHELF, "Dark block = shelf; darker zone background = more pressure"),
    ]
    for index, (color, label_text) in enumerate(legend_items):
        item_y = legend_y + 25 + index * 18
        pygame.draw.rect(screen, color, pygame.Rect(x, item_y + 2, 13, 13), border_radius=2)
        draw_text(screen, small_font, label_text, x + 21, item_y, TEXT)

    draw_text(screen, small_font, "Green KPI = Hybrid better; red KPI = Hybrid worse", x, legend_y + 25 + len(legend_items) * 18 + 2, MUTED)
    event_y = legend_y + 25 + len(legend_items) * 18 + 24
    draw_text(screen, heading_font, "LATEST EVENT", x, event_y)
    if runner.event_log:
        message = str(runner.event_log[0].get("message", ""))
        draw_wrapped_text(
            screen,
            small_font,
            message,
            pygame.Rect(x, event_y + 23, width, 45),
            MUTED,
            max_lines=2,
        )
    if situation["replay_finished"]:
        draw_text(screen, small_font, "Footwear replay has reached its last scheduled task.", x, panel_rect.bottom - 24, SURGE)


def main() -> int:
    args = parse_args()
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = SimulationConfig(order_surge_count=args.surge_count)
    runner = ComparisonRunner(
        config,
        persist=True,
        tasks_path=args.tasks,
        manifest_path=args.manifest,
    )

    pygame.init()
    cell_size = 22
    grid_px = config.grid_width * cell_size
    margin = 20
    gap = 24
    panel_width = 450
    origin_y = 70
    left_x = margin
    right_x = left_x + grid_px + gap
    panel_x = right_x + grid_px + 24
    width = panel_x + panel_width + margin
    height = 860
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Adaptive Multi-Robot Warehouse Orchestration")
    clock = pygame.time.Clock()
    panel_rect = pygame.Rect(panel_x, 12, panel_width, height - 38)
    button_rect = pygame.Rect(panel_x + 18, 500, panel_width - 36, 54)

    running = True
    surge_fired = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    runner.paused = not runner.paused
                elif event.key == pygame.K_r:
                    runner.reset()
                    surge_fired = False
                elif event.key == pygame.K_s:
                    runner.trigger_order_surge(args.surge_zone, args.surge_count)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    runner.trigger_order_surge(args.surge_zone, args.surge_count)

        if args.surge_tick > 0 and not surge_fired and runner.tick_count + 1 == args.surge_tick:
            runner.trigger_order_surge(args.surge_zone, args.surge_count)
            surge_fired = True

        runner.step()
        screen.fill(BACKGROUND)
        draw_engine(
            screen,
            runner.reactive,
            origin_x=left_x,
            origin_y=origin_y,
            cell_size=cell_size,
            title="REACTIVE BASELINE",
            subtitle="Moves only after a Footwear picking task is released",
        )
        draw_engine(
            screen,
            runner.hybrid,
            origin_x=right_x,
            origin_y=origin_y,
            cell_size=cell_size,
            title="ADAPTIVE HYBRID",
            subtitle="Uses predictive repositioning only when thresholds say it is worthwhile",
            show_policy=True,
        )
        draw_panel(screen, runner, panel_rect, button_rect)
        help_font = pygame.font.SysFont("arial", 12)
        draw_text(screen, help_font, "SPACE pause | S surge | R reset | ESC quit", margin, height - 23, MUTED)
        pygame.display.flip()
        clock.tick(config.fps)

        if args.max_ticks > 0 and runner.tick_count >= args.max_ticks:
            running = False

    runner.store.write_state(runner.snapshot())
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
