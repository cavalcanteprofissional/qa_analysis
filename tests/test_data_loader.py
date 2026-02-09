import pandas as pd
from pathlib import Path
from src.data_loader import ShardLoader


def test_discover_shards():
    sl = ShardLoader("data/shards")
    assert isinstance(sl.available_shards, list)


def test_select_all():
    sl = ShardLoader("data/shards")
    sel = sl.select_shards("all")
    assert len(sel) == len(sl.available_shards)
