import pandas as pd

from smart_warehouse.features import FEATURE_COLUMNS, TARGET_COLUMN, build_training_table


def test_training_table_has_required_columns() -> None:
    tasks = pd.DataFrame(
        [
            {"absolute_minute": 1, "zone": "A"},
            {"absolute_minute": 7, "zone": "A"},
            {"absolute_minute": 11, "zone": "B"},
        ]
    )
    table = build_training_table(tasks, total_minutes=30)
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(table.columns)
    assert table[TARGET_COLUMN].min() >= 0
