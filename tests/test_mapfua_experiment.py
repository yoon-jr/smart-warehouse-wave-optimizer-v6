from scripts.run_mapfua_experiment import choose_start_ticks
from smart_warehouse.footwear_replay import FootwearTaskReplay


def test_seeded_windows_are_reproducible_and_within_test_split(prepared_footwear_paths) -> None:
    replay = FootwearTaskReplay(
        prepared_footwear_paths["tasks"],
        manifest_path=prepared_footwear_paths["manifest"],
    )
    first = choose_start_ticks(
        replay,
        split_name="test",
        ticks=4,
        seeds=[42, 43, 44],
        fixed_start_tick=None,
    )
    second = choose_start_ticks(
        replay,
        split_name="test",
        ticks=4,
        seeds=[42, 43, 44],
        fixed_start_tick=None,
    )
    assert first == second
    assert set(first) == {42, 43, 44}
    assert all(replay.start_tick <= value <= replay.max_tick for value in first.values())


def test_fixed_window_is_shared_across_policies_and_seeds(prepared_footwear_paths) -> None:
    replay = FootwearTaskReplay(prepared_footwear_paths["tasks"])
    starts = choose_start_ticks(
        replay,
        split_name="test",
        ticks=4,
        seeds=[1, 2, 3],
        fixed_start_tick=replay.start_tick,
    )
    assert set(starts.values()) == {replay.start_tick}
